from transformers import AutoTokenizer, AutoConfig
import torch

from llama_prompt import GenerativePromptLlama

def strip_uri(uris, prefix="<http://d-nb.info/gnd/", suffix=">"):
    uris = uris.split()
    return [uri.removeprefix(prefix).removesuffix(suffix)
            for uri in uris]

def get_pref_label(graph, node_id):
    node_data = graph.nodes[node_id]
    prefered_name = node_data.get("pref", None)
    if prefered_name is not None:
        return list(prefered_name)[0]
    else:
        return None

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
    if k == 0:
        return neighbors
    for neighbor in graph.neighbors(node_id):
        if relation is not None:
            rel_type = get_relation_type(graph, node_id, neighbor)
            if rel_type != relation:
                continue
        neighbors.add(neighbor)
        neighbors.update(k_hop_neighbors(graph, neighbor, k-1))
    return neighbors

def get_label_mapping(graph):
    label_mapping = {}
    label_strings = []
    for idx, node_id in enumerate(graph.nodes):
        label = get_pref_label(graph, node_id)
        if label is not None:
            label_strings.append(label)
            label_mapping[idx] = node_id
    return label_strings, label_mapping

def precision_at_k(y_true, y_pred, k):
    y_pred = y_pred[:k]
    correct = len(set(y_true).intersection(set(y_pred)))
    return correct / k

def recall_at_k(y_true, y_pred, k):
    y_pred = y_pred[:k]
    correct = len(set(y_true).intersection(set(y_pred)))
    return correct / len(y_true)

def f1_at_k(y_true, y_pred, k):
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def tokenize(record, tokenizer, max_length=None, sep_token=", ", suffix="", prefix=""):

    prefix_tok_out = tokenizer(prefix, add_special_tokens=False)
    suffix_tok_out = tokenizer(suffix, add_special_tokens=False)
    target_tok_out = tokenizer(sep_token.join(record["label_list"]), add_special_tokens=False)

    # Truncate the prompt if it's too long. Subtract 2 for the [BOS] and [EOS] tokens.
    trunc_len = max_length - len(prefix_tok_out["input_ids"]) - len(suffix_tok_out["input_ids"]) - len(target_tok_out["input_ids"]) - 2
    if trunc_len <= 0:
        raise ValueError("Input is too long. Increase max_length.")
    prompt_tok_out = tokenizer(record["title"], add_special_tokens=False, truncation=True, max_length=trunc_len)
    
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
    return {"input_ids": full_ids, "labels": labels, "attention_mask": full_attention_mask}

def init_prompt_model(model_name, prompt_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>") # Ensure padding token is the same as EOS token
    model = GenerativePromptLlama.from_pretrained(model_name, prompt_config=prompt_config)
    for param in model.parameters():
        param.requires_grad = False
    model.model.add_prompt()
    return model, tokenizer


