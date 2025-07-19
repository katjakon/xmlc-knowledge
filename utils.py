import re

import networkx as nx
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from safetensors import safe_open

from llama_prompt import GenerativePromptLlama

PAD_TOKEN = "<|finetune_right_pad_id|>"
EOT_TOKEN = "<|eot_id|>"
BATCH_KEYS = ["input_ids", "attention_mask", "labels", "seq_lengths", "context_ids", "context_lengths"]
SEP_TOKEN = ";"

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

def jaccard_similarity(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    correct = y_true.intersection(y_pred)
    return len(correct) / (len(y_pred) + len(y_true))

def inverse_distance_weight(graph, gold_node, predicted_node):
    """
    Calculate graph distance weight between two nodes in a graph.
    The weight is inversely proportional to the shortest path distance between the nodes.

    Args:
        graph (networkx.Graph): The graph containing the nodes.
        gold_node (str): The node representing the gold standard.
        predicted_node (str): The node representing the predicted label.
    
    Returns:
        float: The weight based on the distance between the nodes.
    """
    weight = 0.0
    # Compute shortest path distance
    if gold_node == predicted_node:
        distance = 0  # Perfect match
    else:
        try:
            if gold_node not in graph or predicted_node not in graph:
                return 0.0
            distance = nx.shortest_path_length(graph, source=gold_node, target=predicted_node)
        except nx.NetworkXNoPath:
            distance = float('inf')  # No path exists

        # Weight inversely proportional to distance
    weight = 1 / (1 + distance) if distance != float('inf') else 0

    # Normalize by the number of predictions
    return weight

def weighted_precision(y_true, y_pred, graph):
    weighted_prec = []
    for p in y_pred:
        max_weight = - float('inf')
        for g in y_true:
            weight = inverse_distance_weight(graph, g, p)
            if weight > max_weight:
                max_weight = weight
        weighted_prec.append(max_weight)
    weighted_prec = sum(weighted_prec) / len(weighted_prec) if weighted_prec else 0
    return weighted_prec

def process_output(text):
    # pattern = r"\d+[.)]"
    # text = re.sub(pattern, "", text)
    sep_tokens = r"[,;-]"
    text = re.split(sep_tokens, text)
    text = [x.strip() for x in text]
    if len(text) == 1:
        text = text[0].split(" ")
    return [keyword for keyword in text if keyword]

def tokenize(batch, tokenizer, suffix="", prefix="", max_length=55):

    texts = [
        f"{prefix}{item['title']}{suffix}"
        for item in batch
        ]

    labels = [f"{SEP_TOKEN} ".join(item['label-names']) for item in batch]

    tokenized_inputs = tokenizer(
        texts,
        padding=False,  # No padding yet
        truncation=True,  
        max_length=max_length,  # Truncate to max_length
        add_special_tokens=False,
        return_attention_mask=False,
    )

    input_ids = [
        [tokenizer.bos_token_id] + input_id
        for input_id in tokenized_inputs['input_ids']
    ]

    input_lengths = torch.tensor([len(input_id) for input_id in input_ids])

    # Tokenize labels
    tokenized_labels = tokenizer(
        labels,
        padding=False,  # No padding yet
        truncation=False,  # Do not truncate
        add_special_tokens=False, 
        return_attention_mask=False,
    )
    label_ids = [
        label_id + [tokenizer.eos_token_id]
        for label_id in tokenized_labels['input_ids']
    ]

    # Concatenate input IDs and label IDs
    full_ids = [
        torch.tensor(input_id + label_id)
        for input_id, label_id in zip(input_ids, label_ids)
    ]
    full_ids = torch.nn.utils.rnn.pad_sequence(full_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.full_like(full_ids, fill_value=-100)  # Initialize with -100

    for i, length in enumerate(input_lengths):
        labels[i, length:] = full_ids[i, length:]  # Replace target positions with actual values

    attention_mask = (full_ids != tokenizer.pad_token_id).long()
    labels[~attention_mask.bool()] = -100  # Set padding positions in labels to -100
    return {"input_ids": full_ids, "labels": labels, "attention_mask": attention_mask, "seq_lengths": input_lengths}

def inference_tokenize(batch, tokenizer, suffix="", prefix=""):
    texts = f"{prefix}{batch['title']}{suffix}"
    tokenized_inputs = tokenizer(
        texts,
        padding=False,  # No padding yet
        truncation=False,  # Do not truncate
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = torch.tensor([tokenizer.bos_token_id] + tokenized_inputs['input_ids'])
    input_lengths = torch.tensor(len(input_ids))
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "seq_lengths": input_lengths}

def tokenize_context(batch, tokenizer):
    strings = [SEP_TOKEN.join(item) for item in batch["context_str"]]
    tokenized_context = tokenizer(
        strings,
        padding=False,  # No padding yet
        truncation=False,  # Do not truncate
        add_special_tokens=False,
        return_attention_mask=False,
    )
    context_length = torch.tensor([len(context) for context in tokenized_context['input_ids']])
    context_ids = torch.nn.utils.rnn.pad_sequence(tokenized_context["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
    return {"context_ids": context_ids, "context_lengths": context_length}

def init_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    return tokenizer

def init_prompt_model(model_name, prompt_config, tune_lm_head=True):
    tokenizer = init_tokenizer(model_name)
    model = GenerativePromptLlama.from_pretrained(model_name, prompt_config=prompt_config)
    for param in model.parameters():
        param.requires_grad = False
    if tune_lm_head: 
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
        title_batch = {k: torch.tensor(v).to(device).unsqueeze(0) for k, v in title_batch.items() if k in BATCH_KEYS}
        with torch.no_grad():
            if isinstance(model, torch.nn.DataParallel):
                gen_model = model.module
            else:
                gen_model = model
            generated_ids = gen_model.generate(
                **title_batch,
                )
        len_input = len(title_batch["input_ids"][0])
        generated_ids = generated_ids[0][len_input:] 
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predictions.append(generated_text)
    return predictions

def map_labels(prediction_list, index, retriever, label_mapping, k=1):
    mapped_predictions = []
    for pred_list in tqdm(prediction_list, desc="Mapping predictions to GND labels"):
        current_mapping = []
        for pred in pred_list:
            distance, idns = retriever.retrieve(
                mapping=label_mapping,
                index=index,
                texts=[pred],
                top_k=k)
            idn_sim = zip(idns[0], distance[0])
            current_mapping.extend(idn_sim)
        current_mapping = sorted(current_mapping, key=lambda x: x[1])
        current_mapping = [x[0] for x in current_mapping]
        current_mapping = list(set(current_mapping))
        mapped_predictions.append(current_mapping)
    return mapped_predictions