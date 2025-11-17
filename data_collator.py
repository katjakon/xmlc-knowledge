from typing import Any
import torch
import torch_geometric as pyg
from utils import SEP_TOKEN
from prompt_str import SUFFIX_PROMPT, PREFIX_PROMPT, SYSTEM_PROMPT_EXAMPLE
from retriever import Retriever

def format_string(input_text: str, label_list: list, prefix: str = "", suffix: str = "") -> str:
    """Formats a string with a prefix and suffix."""
    input_str = f"{prefix}{input_text}{suffix}"
    label_str = f"{SEP_TOKEN} ".join(label_list)
    return input_str, label_str


class DataCollator:

    def __init__(self, tokenizer, graph, device, retriever=None, use_context = False, top_k=3, hops=0, graph_based=False) -> None:
        self.tokenizer = tokenizer
        self.use_context = use_context  # Whether to use context
        self.top_k = top_k  # Number of neighbors to retrieve
        self.hops = hops # Number of neighbors to get from top k retrieved neighbors.
        self.device = device
        self.graph = graph  
        self.retriever = retriever
        self.graph_based = graph_based
        self.graph_data = None
    
    def add_graph_data(self, idn2idx, idx2idn, pyg_data):
        self.graph_data = {
            "idn2idx": idn2idx, 
            "idx2idn": idx2idn,
            "data": pyg_data

        }
        return self.graph_data

    def tokenize(self, batch, suffix="", prefix="", max_length=55):

        texts = [
            f"{prefix}{item['title']}{suffix}"
            for item in batch
            ]

        labels = [f"{SEP_TOKEN} ".join(item['label-names']) for item in batch]

        tokenized_inputs = self.tokenizer(
            texts,
            padding=False,  # No padding yet
            truncation=True,  
            max_length=max_length,  # Truncate to max_length
            add_special_tokens=False,
            return_attention_mask=False,
        )

        input_ids = [
            [self.tokenizer.bos_token_id] + input_id
            for input_id in tokenized_inputs['input_ids']
        ]

        input_lengths = torch.tensor([len(input_id) for input_id in input_ids])

        # Tokenize labels
        tokenized_labels = self.tokenizer(
            labels,
            padding=False,  # No padding yet
            truncation=False,  # Do not truncate
            add_special_tokens=False, 
            return_attention_mask=False,
        )
        label_ids = [
            label_id + [self.tokenizer.eos_token_id]
            for label_id in tokenized_labels['input_ids']
        ]

        # Concatenate input IDs and label IDs
        full_ids = [
            torch.tensor(input_id + label_id)
            for input_id, label_id in zip(input_ids, label_ids)
        ]
        full_ids = torch.nn.utils.rnn.pad_sequence(full_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.full_like(full_ids, fill_value=-100)  # Initialize with -100

        for i, length in enumerate(input_lengths):
            labels[i, length:] = full_ids[i, length:]  # Replace target positions with actual values

        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()
        labels[~attention_mask.bool()] = -100  # Set padding positions in labels to -100
        if full_ids.shape[0] == 1:
            full_ids = full_ids.squeeze(0)
            labels = labels.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            input_lengths = input_lengths.squeeze(0)
        return {"input_ids": full_ids, "labels": labels, "attention_mask": attention_mask, "seq_lengths": input_lengths}
    
    def tokenize_context(self, batch, max_length=50):
        if not isinstance(batch, list):
            batch = [batch]
        strings = [f"{SEP_TOKEN} ".join(item) for item in batch]
        tokenized_context = self.tokenizer(
            strings,
            padding=False,  # No padding yet
            truncation=True,
            max_length=max_length,  # Truncate to max_length
            add_special_tokens=False,
            return_attention_mask=False,
        )
        input_ids = [torch.tensor(input_id) for input_id in tokenized_context['input_ids']]
        context_length = torch.tensor([len(context) for context in tokenized_context['input_ids']])
        context_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if context_ids.shape[0] == 1:
            context_ids = context_ids.squeeze(0)
        return {"context_ids": context_ids, "context_lengths": context_length}
    
    def inference_tokenize(self, batch, suffix="", prefix=""):
        if len(batch) > 1:
            raise ValueError("Inference tokenization is intended for single-item batches only.")
        batch = batch[0]  # Unpack single-item batch
        texts = f"{prefix}{batch['title']}{suffix}"
        tokenized_inputs = self.tokenizer(
            texts,
            padding=False,  # No padding yet
            truncation=False,  # Do not truncate
            add_special_tokens=False,
            return_attention_mask=False,
        )
        input_ids = torch.tensor([self.tokenizer.bos_token_id] + tokenized_inputs['input_ids'])
        input_lengths = torch.tensor(len(input_ids))
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_lengths": input_lengths
        })
        return batch
    
    def __call__(self, batch_list, inference=False, max_input_len=55, max_context_len=50, keys=None) -> Any:
        if inference:
            batch = self.inference_tokenize(batch_list, suffix=SUFFIX_PROMPT, prefix=PREFIX_PROMPT)
        else:
            batch = self.tokenize(batch_list, suffix=SUFFIX_PROMPT, prefix=PREFIX_PROMPT, max_length=max_input_len)
        if self.use_context:
            if self.retriever is None:
                raise ValueError("Retriever must be provided for context retrieval.")
            context_idns = self.retriever.retrieve_with_neighbors(
                texts=[item["title"] for item in batch_list],
                top_k=self.top_k, 
                k=self.hops
            )
            if self.graph_based is False: # Only use text-based information
                context_str = [
                    [self.graph.pref_label_name(idn) for idn in idn_list]
                    for idn_list in context_idns]
                context_batch = self.tokenize_context(context_str, max_length=max_context_len)
            else:
                if self.graph_data is None: 
                    raise ValueError("No graph data was generated!")
                idn2idx = self.graph_data["idn2idx"]
                graph_data = self.graph_data["data"]
                context_idx = [
                    [idn2idx[idn] for idn in idn_list]
                      for idn_list in context_idns]
                graph_batch = [graph_data.subgraph(torch.tensor(indices)) for indices in context_idx]
                graph_batch = pyg.data.Batch.from_data_list(graph_batch)
                context_batch = {"graph_batch": graph_batch}
            batch.update(context_batch)
        if keys is not None:
            batch = {key: batch[key] for key in keys if key in batch}
        return batch

class GraphDataCollator:

    LABEL = "Schlagwort: "
    NEIGHBORS = "Nachbarn: "
    ALT_NAMES = "Alternative Namen: "

    def __init__(self, tokenizer, config, device, neighbors=True) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.neighbors = neighbors

    def tokenize(self, batch,  max_length=55):
            
            texts = []

            for item in batch:
                txt = ""
                if self.neighbors:
                    if item["neighbors"]:
                        neighbors_text = f"{SEP_TOKEN} ".join(item["neighbors"])
                        neighbors_texts = f"{self.NEIGHBORS}{neighbors_text}. "
                        txt += neighbors_texts
                else:
                    txt += f"{self.LABEL}: {item['label']} "
                    if item["alt-names"]:
                        alt_names_text = f"{SEP_TOKEN} ".join(item["alt-names"])
                        alt_names_texts = f"{self.ALT_NAMES}{alt_names_text}. "
                        txt += alt_names_texts
                    else:
                        txt += f"{self.LABEL}: {item['label']} "
                txt += self.LABEL
                texts.append(txt)
            labels = [item['label'] for item in batch]

            tokenized_inputs = self.tokenizer(
                texts,
                padding=False,  # No padding yet
                truncation=True,  
                max_length=max_length,  # Truncate to max_length
                add_special_tokens=False,
                return_attention_mask=False,
            )

            input_ids = [
                [self.tokenizer.bos_token_id] + input_id
                for input_id in tokenized_inputs['input_ids']
            ]

            input_lengths = torch.tensor([len(input_id) for input_id in input_ids])
            # Tokenize labels
            tokenized_labels = self.tokenizer(
                labels,
                padding=False,  # No padding yet
                truncation=False,  # Do not truncate
                add_special_tokens=False, 
                return_attention_mask=False,
            )
            label_ids = [
                label_id + [self.tokenizer.eos_token_id]
                for label_id in tokenized_labels['input_ids']
            ]

            # Concatenate input IDs and label IDs
            full_ids = [
                torch.tensor(input_id + label_id)
                for input_id, label_id in zip(input_ids, label_ids)
            ]
            full_ids = torch.nn.utils.rnn.pad_sequence(full_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.full_like(full_ids, fill_value=-100)  # Initialize with -100

            for i, length in enumerate(input_lengths):
                labels[i, length:] = full_ids[i, length:]  # Replace target positions with actual values

            attention_mask = (full_ids != self.tokenizer.pad_token_id).long()
            labels[~attention_mask.bool()] = -100  # Set padding positions in labels to -100
            if full_ids.shape[0] == 1:
                full_ids = full_ids.squeeze(0)
                labels = labels.squeeze(0)
                attention_mask = attention_mask.squeeze(0)
                input_lengths = input_lengths.squeeze(0)
            return {"input_ids": full_ids, "labels": labels, "attention_mask": attention_mask, "seq_lengths": input_lengths}

    def __call__(self, batch_list, max_length=100) -> Any:
        batch = self.tokenize(batch_list, max_length=max_length)
        return batch



