import re

from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from safetensors import safe_open

from llama_prompt import GenerativePromptLlama

PAD_TOKEN = "<|finetune_right_pad_id|>"
EOT_TOKEN = "<|eot_id|>"

def strip_uri(uris, prefix="<http://d-nb.info/gnd/", suffix=">"):
    uris = uris.split()
    return [uri.removeprefix(prefix).removesuffix(suffix)
            for uri in uris]

def get_pref_label(graph, node_id):
    node_data = graph.nodes.get(node_id)
    if node_data is None:
        return None
    prefered_name = node_data.get("pref", None)
    if prefered_name is not None:
        return list(prefered_name)[0]
    else:
        return None

def get_alt_labels(graph, node_id):
    node_data = graph.nodes.get(node_id)
    if node_data is None:
        return []
    alternative_names = node_data.get("alt", None)
    if alternative_names is not None:
        return list(alternative_names)
    else:
        return []

def get_relation_type(graph, head, tail):
    edge_data = graph.edges.get((head, tail))
    if edge_data is not None:
        return edge_data.get("relation")
    else:
        return None

def get_node_type(graph, node_id):
    node_data = graph.nodes.get(node_id)
    if node_data is not None:
        return node_data.get("type")
    else:
        return None

def k_hop_neighbors(graph, node_id, k, relation=None):
    neighbors = set()
    if k == 0 or node_id not in graph.nodes:
        return neighbors
    for neighbor in graph.neighbors(node_id):
        if relation is not None:
            rel_type = get_relation_type(graph, node_id, neighbor)
            if rel_type != relation:
                continue
        neighbors.add(neighbor)
        neighbors.update(k_hop_neighbors(graph, neighbor, k-1))
    return neighbors

def get_label_mapping(graph, use_alt_labels=False):
    label_mapping = {}
    label_strings = []
    counter = 0
    for node_id in graph.nodes:
        label = get_pref_label(graph, node_id)
        if label is not None:
            label_strings.append(label)
            label_mapping[counter] = node_id
            counter += 1
        if use_alt_labels:
            alt_labels = get_alt_labels(graph, node_id)
            for alt_label in alt_labels:
                label_strings.append(alt_label)
                label_mapping[counter] = node_id
                counter += 1
    return label_strings, label_mapping

def get_title_mapping(title_ds):
    title_mapping = {}
    title_strings = []
    for idx, row in enumerate(title_ds):
        title = row["title"]
        labels = row["label-ids"]
        title_strings.append(title)
        title_mapping[idx] = labels
    return title_strings, title_mapping

def precision_at_k(y_true, y_pred, k=None):
    if k is not None:
        y_pred = y_pred[:k]
    correct = len(set(y_true).intersection(set(y_pred)))
    return correct / len(y_pred) if len(y_pred) > 0 else 0

def recall_at_k(y_true, y_pred, k=None):
    if k is not None:
        y_pred = y_pred[:k]
    correct = len(set(y_true).intersection(set(y_pred)))
    return correct / len(y_true)

def f1_at_k(y_true, y_pred, k=None):
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def process_output(text):
    pattern = r"\d+[.)]"
    text = re.sub(pattern, "", text)
    sep_tokens = r"[,;-]"
    text = re.split(sep_tokens, text)
    text = [x.strip() for x in text]
    if len(text) == 1:
        text = text[0].split(" ")
    return [keyword for keyword in text if keyword]

def tokenize(record, tokenizer, max_length=512, suffix="", prefix=""):

    label_str = record["label-string"]
    title = record["title"]
    prefix_tok_out = tokenizer(prefix, add_special_tokens=False)
    suffix_tok_out = tokenizer(suffix, add_special_tokens=False)
    prompt_tok_out = tokenizer(record["title"], add_special_tokens=False, truncation=True, max_length=75)

    # Truncate the prompt if it's too long. Subtract 2 for the [BOS] and [EOS] tokens.
    trunc_len = max_length - len(prefix_tok_out["input_ids"]) - len(suffix_tok_out["input_ids"]) - len(prompt_tok_out["input_ids"]) - 2
    if trunc_len <= 0:
        raise ValueError(f"Input '{title}' is too long ({-trunc_len}). Increase max_length.")
    target_tok_out = tokenizer(label_str, add_special_tokens=False, truncation=True, max_length=trunc_len)

    # Add special tokens
    input_ids = [tokenizer.bos_token_id] + prefix_tok_out["input_ids"] + prompt_tok_out["input_ids"] + suffix_tok_out["input_ids"]
    target_ids = target_tok_out["input_ids"] + [tokenizer.eos_token_id]

    len_text = len(input_ids) + len(target_ids)
    padding_length = max_length - len_text
    full_ids = input_ids + target_ids
    full_ids = full_ids + [tokenizer.pad_token_id] * padding_length # Pad to max_length.
    labels = [-100] * len(input_ids) + target_ids + [-100] * padding_length # Only compute loss on target tokens.
    full_attention_mask = [1] * len_text + [0] * padding_length # Mask padding tokens.

    # Convert to tensors
    full_ids = torch.tensor(full_ids)
    labels = torch.tensor(labels)
    full_attention_mask = torch.tensor(full_attention_mask)
    return {"input_ids": full_ids, "labels": labels, "attention_mask": full_attention_mask, "seq_lengths": len(input_ids)}

def inference_tokenize(record, tokenizer, max_length=512, suffix="", prefix=""):

    prefix_tok_out = tokenizer(prefix, add_special_tokens=False)
    suffix_tok_out = tokenizer(suffix, add_special_tokens=False)

    # Truncate the prompt if it's too long. Subtract 2 for the [BOS] and [EOS] tokens.
    trunc_len = max_length - len(prefix_tok_out["input_ids"]) - len(suffix_tok_out["input_ids"]) - 2
    if trunc_len <= 0:
        raise ValueError("Input is too long. Increase max_length.")
    prompt_tok_out = tokenizer(record["title"], add_special_tokens=False, truncation=True, max_length=trunc_len)
    
    # Add special tokens
    input_ids = [tokenizer.bos_token_id] + prefix_tok_out["input_ids"] + prompt_tok_out["input_ids"] + suffix_tok_out["input_ids"]

    len_text = len(input_ids) 
    full_ids = input_ids 
    full_attention_mask = [1] * len_text 

    # Convert to tensors
    full_ids = torch.tensor(full_ids)
    full_attention_mask = torch.tensor(full_attention_mask)
    return {"input_ids": full_ids, "attention_mask": full_attention_mask, "seq_lengths": len(input_ids)}


def init_prompt_model(model_name, prompt_config, tune_lm_head=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN) 
    model = GenerativePromptLlama.from_pretrained(model_name, prompt_config=prompt_config)
    for param in model.parameters():
        param.requires_grad = False
    if tune_lm_head: 
        for param in model.lm_head.parameters():
            param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True
    model.model.add_prompt()
    return model, tokenizer

def load_model(checkpoint_path, config, device, data_parallel=True):
    prompt_config = config["prompt_config"]
    model_name = config["model_name"]
    model, tokenizer = init_prompt_model(model_name, prompt_config)
    tensors = {}
    with safe_open(checkpoint_path, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    # Remove prefix "module." from keys.
    tensors = {k.removeprefix("module."): v for k, v in tensors.items()}
    incompatible_keys = model.load_state_dict(tensors, strict=False)
    # Missing keys are expected since we only tune a fraction of the model.
    # Unexpected keys should be reported.
    if len(incompatible_keys.unexpected_keys) > 0: 
        raise ValueError(f"Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}")
    if data_parallel:
        model = torch.nn.DataParallel(model)
    return model, tokenizer

def generate_predictions(model, tokenizer, dataset, device="cuda"):
    model.eval()
    predictions = []
    for title_batch in tqdm(dataset, desc="Generating labels..."):
        input_ids = torch.tensor(title_batch["input_ids"]).to(device).unsqueeze(0)
        attention_mask = torch.tensor(title_batch["attention_mask"]).to(device).unsqueeze(0)
        seq_lengths = torch.tensor(title_batch["seq_lengths"]).to(device).unsqueeze(0)
        with torch.no_grad():
            if isinstance(model, torch.nn.DataParallel):
                gen_model = model.module
            else:
                gen_model = model
            generated_ids = gen_model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                seq_lengths=seq_lengths
                )
        len_input = len(input_ids[0])
        generated_ids = generated_ids[0][len_input:] 
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predictions.append(generated_text)
    return predictions

def map_labels(prediction_list, index, retriever, label_mapping):
    mapped_predictions = []
    for pred_list in tqdm(prediction_list, desc="Mapping predictions to GND labels"):
        current_mapping = []
        for pred in pred_list:
            distance, idns = retriever.retrieve(
                mapping=label_mapping,
                index=index,
                texts=[pred],
                top_k=1)
            idn_sim = zip(idns[0], distance[0])
            current_mapping.extend(idn_sim)
        current_mapping = sorted(current_mapping, key=lambda x: x[1])
        current_mapping = [x[0] for x in current_mapping]
        current_mapping = list(set(current_mapping))
        mapped_predictions.append(current_mapping)
    return mapped_predictions