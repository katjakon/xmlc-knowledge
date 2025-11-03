import argparse
import os
import pickle
import re
import yaml
import random

from datasets import Dataset
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import transformers
from transformers import  pipeline, set_seed
from tqdm import tqdm

from gnd_graph import GNDGraph
from gnd_dataset import GNDDataset
from reranker import BGEReranker
from retriever import Retriever
from utils import process_output, map_labels, load_config
from prompt_str import SYSTEM_PROMPT, USER_PROMPT, FS_PROMPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--result_dir", type=str, help="Path to the result directory.")
parser.add_argument("--split", type=str, help="Split to use for evaluation.", default="test")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--index", type=str, help="Path to the index file.")
parser.add_argument("--mapping", type=str, help="Path to the mapping file.")
parser.add_argument("--n_examples", type=int, help="Number of examples.", default=3)
parser.add_argument(
    "--example-type", 
    type=str,
    help="How to choose few shot examples.", 
    default="title", 
    choices=("title", "label", "random"))
parser.add_argument("--dev", action='store_true', default=False)

arguments = parser.parse_args()
config_path = arguments.config
result_dir = arguments.result_dir
split = arguments.split
index_path = arguments.index
mapping_path = arguments.mapping
example_type = arguments.example_type
n_examples = arguments.n_examples
dev = arguments.dev

set_seed(arguments.seed)

# Load config 
config = load_config(config_path)

exp_name = config["experiment_name"]
model_name = config["model_name"]

print("Loading Graph.")
# Load GND graph
gnd_path  = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)


print("Loading dataset.")
gnd_ds = GNDDataset(
    data_dir=config["dataset_path"],
    gnd_graph=gnd_graph,
    load_from_disk=True,
)
train_ds = gnd_ds["train"]
test_ds = gnd_ds["test"]

if dev:
    train_ds = train_ds.select(range(100))
    test_ds = test_ds.select(range(10))

print("Loading retriever.")
retriever = Retriever(
    retriever_model=config["sentence_transformer_model"],
    graph=gnd_graph,
    device=DEVICE,
)

retriever.load_search_index(
    index_path=index_path,
    mapping_path=mapping_path,
)

# Create mapping from label ids to document index in the training data.
label_doc_dict = {}
for idx, instance in enumerate(train_ds):
    doc_idn = instance["doc_idn"]
    for idn in instance["label-ids"]:
        if idn not in label_doc_dict:
            label_doc_dict[idn] = set()
        label_doc_dict[idn].add(idx)

test_titles = test_ds["title"]
distance, label_idn = retriever.retrieve(
    texts=test_titles,
    top_k=config["context"]["top_k"]
)

for test_inst, l_idns in zip(test_ds, label_idn):
    print(test_inst["title"])
    for i in l_idns:
        fs_set = label_doc_dict.get(i, [])
        for fs in fs_set:
            print(train_ds[fs])



