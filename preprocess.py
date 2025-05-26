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

parser = argparse.ArgumentParser(description="Create a dataset.")
parser.add_argument("--data_dir", type=str, help="Path to the GND dataset directory.")
parser.add_argument("--gnd_graph", type=str, help="Path to the GND graph file (pickle).")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--out_dir", type=str, help="Path to the out directory.")

arguments = parser.parse_args()
data_dir = arguments.data_dir
gnd_graph = arguments.gnd_graph
config_path = arguments.config
out_dir = arguments.out_dir

# Load config 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

context_config = config["context"]
context = context_config["context_type"]
hops = context_config["hops"]
relation = context_config["relation"]
top_k = context_config["top_k"]
mapping_path = context_config["mapping_path"]
index_path = context_config["index_path"]
title_wise = context_config["title_wise"]

gnd_graph = pickle.load(open(gnd_graph, "rb"))
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
    mapping = pickle.load(open(mapping_path, "rb")) 
    index = pickle.load(open(index_path, "rb"))
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
        batch_size=1024,
        )

# Tokenize the datasets
gnd_ds.tokenize_datasets(tokenizer=tokenizer, splits=["train", "validate"])
gnd_ds.inference_tokenize_datasets(tokenizer=tokenizer, splits=["test"])

# Save the datasets
name = f"{context}-context_{hops}-hops_{relation}-relation_{top_k}-k_titlewise-{title_wise}"
output_path = os.path.join(out_dir, name)
gnd_ds.save_to_disk(output_path)