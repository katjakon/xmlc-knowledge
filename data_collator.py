from typing import Any
import torch
from utils import SEP_TOKEN
from prompt_str import SUFFIX_PROMPT, PREFIX_PROMPT
from retriever import Retriever

class DataCollator:

    def __init__(self, tokenizer, graph, config, device):
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.graph = graph  
        self.retriever = Retriever(
            retriever_model=self.config["sentence_transformer_model"],
            graph=self.graph,
            device=self.device
        )
        if config["context"].get("index_path") is not None and config["context"].get("mapping_path") is not None:
            self.load_search_index()
    
    def load_search_index(self):
        mapping_path = self.config["context"]["mapping_path"]
        index_path = self.config["context"]["index_path"]
        self.retriever.load_search_index(
            mapping_path=mapping_path,
            index_path=index_path
        )

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
    
    def __call__(self, batch_list, inference=False) -> Any:
        if inference:
            batch = self.inference_tokenize(batch_list, suffix=SUFFIX_PROMPT, prefix=PREFIX_PROMPT)
        else:
            batch = self.tokenize(batch_list, suffix=SUFFIX_PROMPT, prefix=PREFIX_PROMPT)
        if self.config["context"].get("context_type") is not None:
            context_idns = self.retriever.retrieve_with_neighbors(
                texts= [item["title"] for item in batch_list],
                top_k=self.config["context"]["top_k"],
                batch_size=len(batch_list),
            )
            context_str = [
                [self.graph.pref_label_name(idn) for idn in idn_list]
                for idn_list in context_idns]
            context_batch = self.tokenize_context(context_str)
            batch.update(context_batch)
        return batch



