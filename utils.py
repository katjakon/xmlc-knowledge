import re

import networkx as nx
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from safetensors import safe_open
from sentence_transformers import SentenceTransformer

from llama_prompt import GenerativePromptLlama

PAD_TOKEN = "<|finetune_right_pad_id|>"
EOT_TOKEN = "<|eot_id|>"
BATCH_KEYS = ["input_ids", "attention_mask", "labels", "seq_lengths", "context_ids", "context_lengths", "graph_batch"]
SEP_TOKEN = ";"

def strip_uri(uris, prefix="<http://d-nb.info/gnd/", suffix=">"):
    uris = uris.split()
    return [uri.removeprefix(prefix).removesuffix(suffix)
            for uri in uris]

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
    sep_tokens = r"[,;-]"
    text = re.split(sep_tokens, text)
    text = [x.strip() for x in text]
    if len(text) == 1:
        text = text[0].split(" ")
    return [keyword for keyword in text if keyword]

def init_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    return tokenizer

def init_prompt_model(model_name, prompt_config, tune_lm_head=True, embeddings=None):
    tokenizer = init_tokenizer(model_name)
    model = GenerativePromptLlama.from_pretrained(model_name, prompt_config=prompt_config)
    for param in model.parameters():
        param.requires_grad = False
    if tune_lm_head: 
        for param in model.lm_head.parameters():
            param.requires_grad = True
    model.model.add_prompt(embeddings=embeddings)
    return model, tokenizer

def load_model(checkpoint_path, config, device, data_parallel=True, load=None):
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        config (dict): Configuration dictionary containing model and prompt configurations.
        device (str): Device to load the model on, e.g., "cuda" or "cpu".
        data_parallel (bool): Whether to wrap the model in DataParallel. Default is True.
        load (list, optional): List of keys to load from the checkpoint. If None, all tensors are loaded.
    Returns:
        model (torch.nn.Module): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
    """
    prompt_config = config["prompt_config"]
    model_name = config["model_name"]
    model, tokenizer = init_prompt_model(model_name, prompt_config)
    tensors = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        for k in f.keys():
            if load is not None: # Only load specified tensors
                for load_key in load:
                    if load_key in k:
                        tensors[k] = f.get_tensor(k)
            else: # load all tensors if load is None
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
    model.to(device)
    return model, tokenizer

def generate_predictions(model, tokenizer, dataset, device="cuda", num_beams=1, temperature=None, top_p=None, do_sample=False):
    model.eval()
    predictions = []
    for title_batch in tqdm(dataset, desc="Generating labels..."):
        title_batch = {k: v.to(device) for k, v in title_batch.items() if k in BATCH_KEYS}
        # .unsqueeze(0)
        for k, v in title_batch.items():
            if isinstance(v, torch.Tensor):
                title_batch[k] = v.unsqueeze(0)
        with torch.no_grad():
            if isinstance(model, torch.nn.DataParallel):
                gen_model = model.module
            else:
                gen_model = model
            generated_ids = gen_model.generate(
                **title_batch,
                temperature=temperature,
                num_beams=num_beams,
                top_p=top_p,
                do_sample=do_sample,
                )
        len_input = len(title_batch["input_ids"][0])
        generated_ids = generated_ids[0][len_input:] 
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predictions.append(generated_text)
    return predictions

def map_labels(prediction_list, retriever, k=1):
    mapped_predictions = []
    for pred_list in tqdm(prediction_list, desc="Mapping predictions to GND labels"):
        current_mapping = []
        for pred in pred_list:
            distance, idns = retriever.retrieve(
                texts=[pred],
                top_k=k)
            idn_sim = zip(idns[0], distance[0])
            current_mapping.extend(idn_sim)
        current_mapping = sorted(current_mapping, key=lambda x: x[1])
        current_mapping = [x[0] for x in current_mapping]
        current_mapping = list(set(current_mapping))
        mapped_predictions.append(current_mapping)
    return mapped_predictions

def generate_graph_data(label_mapping_path, graph):
    """
    Generate data needed for GNN prompt generators.

    Args:
        label_mapping_path (str): Path to arrow file which contains mapping.
        graph (networkx.DiGraph): Graph which contains label relation
    
    Returns:
        3-Tuple with idn to index mapping, index to idn mapping and pytorch geometric data object.
    """
    df = pd.read_feather(label_mapping_path)
    idx2idn, idn2idx = {}, {}
    # Generate mappings from index to idn and reversed.
    for idx, idn in zip(df["index"], df["idn"]):
        idx2idn[idx] = idn
        idn2idx[idn] = idx
    # Create edge format for pyg data.
    head, tail = [], []
    for index, idn in idx2idn.items():
        neighbors = graph.neighbors(idn)
        neighbors_idx = [idn2idx[n_idn] for n_idn in neighbors]
        for n_idx in neighbors_idx:
            head.append(index)
            tail.append(n_idx)
    edge_index = torch.tensor([head, tail])
    # Node features are indices for mapping to embeddings later.
    x = torch.tensor(list(idx2idn.keys()))
    data = Data(x=x, edge_index=edge_index)
    return idn2idx, idx2idn, data

def get_label_embeddings(mapping_df, prompt_config, kind="random", sentence_transformer_model=None, path=None, device=None, freeze=False):
    kinds = {"random", "retriever", "from_file"}
    if kind not in kinds:
        raise ValueError(f"kind needs to be one of {kinds}. Current value: kind={kind}")
    if kind == "random":
        dim = prompt_config["kge_size"]
        label_embeddings = torch.rand((label_df.shape[0], dim))
    elif kind == "retriever":
        if sentence_transformer_model is None:
            raise ValueError(f"Need to provide retriever model for kind={kind}")
        sentence_transformer_model = SentenceTransformer(sentence_transformer_model, device=device)
        label_strings = mapping_df["strings"]
        label_embeddings =  sentence_transformer_model.encode(
            label_strings, 
            batch_size=1024,
            show_progress_bar=True,
            convert_to_tensor=True)
    elif kind == "from_file":
        pass
    with torch.inference_mode():
        label_embeddings = torch.nn.Embedding.from_pretrained(label_embeddings, freeze=freeze)
    return label_embeddings


