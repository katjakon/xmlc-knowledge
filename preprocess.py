import argparse
import os
import pickle
import re
import yaml

import pandas as pd
import torch

from gnd_dataset import GNDDataset
from retriever import Retriever
from utils import init_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--data_dir", type=str, help="Path to the GND dataset directory.")
parser.add_argument("--gnd_graph", type=str, help="Path to the GND graph file (pickle).")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--out_dir", type=str, help="Path to the out directory.")
parser.add_argument("--context", default=None, help="Context added to the dataset, choose from text or graph", choices=["text", "graph"])
parser.add_argument("--hops", default=0, type=int, help="Number of hops to use when adding context.")
parser.add_argument("--relation", default=None, type=str, help="Relations to use when adding context.")
parser.add_argument("--top_k", default=1, type=int, help="Number of top k neighbors to use when adding context.")
parser.add_argument("--mapping", type=str, help="Path to the label mapping file.")
parser.add_argument("--index", type=str, help="Path to the index file.")
parser.add_argument("--title_wise", action="store_true", help="Use title wise context.")

arguments = parser.parse_args()
data_dir = arguments.data_dir
gnd_graph = arguments.gnd_graph
config_path = arguments.config
out_dir = arguments.out_dir
context = arguments.context
hops = arguments.hops
relation = arguments.relation
top_k = arguments.top_k
mapping_path = arguments.mapping
index_path = arguments.index
title_wise = arguments.title_wise

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

gnd_graph = pickle.load(open(gnd_graph, "rb"))
mapping = pickle.load(open(mapping_path, "rb")) 
index = pickle.load(open(index_path, "rb"))
tokenizer = init_tokenizer(config["model_name"])

retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    retriever_model=retriever_model,
    device=DEVICE,
)

gnd_ds = GNDDataset(
    data_dir=data_dir,
    gnd_graph=gnd_graph,
    config=config
)

if context is not None:
    gnd_ds.add_context(
        retriever=retriever,
        tokenizer=tokenizer,
        index=index,
        mapping=mapping,
        hops=hops,
        relation=relation,
        k=top_k,
        context_type=context,
        use_title_wise=title_wise,
        )

# TODO: Tokenization with context?
# Tokenize the datasets
gnd_ds.tokenize_datasets(tokenizer=tokenizer, splits=["train", "validate"])
gnd_ds.inference_tokenize_datasets(tokenizer=tokenizer, splits=["test"])

# Save the datasets
name = f"{context}-context_{hops}-hops_{relation}-relation_{top_k}-k_titlewise-{title_wise}"
output_path = os.path.join(out_dir, name)
gnd_ds.save_to_disk(output_path)