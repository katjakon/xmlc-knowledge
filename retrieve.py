import argparse
import os
import pickle
import re
import yaml

import pandas as pd
import torch

from gnd_graph import GNDGraph
from gnd_dataset import GNDDataset
from reranker import BGEReranker
from retriever import Retriever
from utils import get_title_mapping, load_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Train a model on the GND dataset.")
parser.add_argument("--config", type=str, help="Path to the configuration file.")
parser.add_argument("--result_dir", type=str, help="Path to the result directory.")
parser.add_argument("--alt_labels", type=bool, default=False, help="Use alternative labels for mapping.")
parser.add_argument("--dev", action='store_true', default=False)
arguments = parser.parse_args()
config_path = arguments.config
result_dir = arguments.result_dir
dev = arguments.dev

# Load config 
config = load_config(config_path)

index_path = config["context"]["index_path"]
mapping_path = config["context"]["mapping_path"]
hops = config["context"]["hops"]
top_k = config["context"]["top_k"]
relation = config["context"]["relation"]
use_title_wise = config["context"]["title_wise"]
exp_name = config["experiment_name"]
batch_size = 1024

result_dir = os.path.join(result_dir, exp_name)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Load GND graph
gnd_path  = config["graph_path"]
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

data_dir = config["dataset_path"]
# Load GND dataset
gnd_ds = GNDDataset(
    data_dir=data_dir,
    gnd_graph=gnd_graph,
    config=config,
    load_from_disk=True,
)
test_ds = gnd_ds["test"]
train_ds = gnd_ds["train"]

if dev:
    test_ds = test_ds.select(range(10))
    train_ds = train_ds.select(range(100))

retriever_model = config["sentence_transformer_model"]
retriever = Retriever(
    graph=gnd_graph,
    retriever_model=retriever_model,
    device=DEVICE,
)

if index_path is not None and mapping_path is not None:
    retriever.load_search_index(
    index_path=index_path,
    mapping_path=mapping_path,
)
else:
    if use_title_wise:
        strings, mapping = get_title_mapping(train_ds)
        retriever.fit_title_wise(title_strings=strings, title_mapping=mapping)
    else:
        retriever.fit(labels=strings, batch_size=batch_size)

if hops is not None and hops > 0:
    idns = retriever.retrieve_with_neighbors(
        texts=test_ds["title"],
        k=hops,
        top_k=top_k,
        batch_size=batch_size,
        relation=relation,
        title_wise=use_title_wise,
    )
else:
    sim, idns = retriever.retrieve(
        texts=test_ds["title"],
        top_k=top_k,
        batch_size=batch_size,
        title_wise=use_title_wise,
        )
pred_df = pd.DataFrame(
    {
        "predictions": idns,
        "label-ids": test_ds["label-ids"],
        "title": test_ds["title"],
        "doc_idn": test_ds["doc_idn"]
    }
)

reranker = BGEReranker("BAAI/bge-reranker-v2-m3", device=DEVICE)
pred_df = reranker.rerank(
    pred_df,
    gnd_graph,
    bs=200
)

pred_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)
