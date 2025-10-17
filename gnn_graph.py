import argparse
import pickle
import os
import json
from torch.utils.data import DataLoader

from networkx import is_path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, GAT
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd

from default_config import default_config
from data_collator import DataCollator
from gnd_graph import GNDGraph
from gnd_dataset import GNDDataset
from retriever import Retriever
from utils import init_tokenizer
from prompt_generators import GraphContextPromptGenerator

device = "cuda"
gnd_path  = "gnd/gnd.pickle"
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)
data_path = "dataset"
ds = GNDDataset(
    data_dir=data_path,
    gnd_graph=gnd_graph, 
    load_from_disk=True,
)

df = pd.read_feather("mapping/label_mapping.feather")
label_strings = list(df["strings"])
idx2idn, idn2idx = {}, {}
for idx, idn in zip(df["index"], df["idn"]):
    idx2idn[idx] = idn
    idn2idx[idn] = idx

tokenizer = init_tokenizer("meta-llama/Llama-3.2-3B")
retriever = Retriever("BAAI/bge-m3", graph=gnd_graph)
retriever.fit(batch_size=1024)
dim = retriever.retriever.get_sentence_embedding_dimension()

# label_embeddings = retriever.retriever.encode(
#     label_strings, 
#     batch_size=1024,
#     show_progress_bar=True,
#     convert_to_tensor=True)
label_embeddings = torch.rand((len(label_strings), dim))

collator = DataCollator(
    tokenizer=tokenizer, 
    graph=gnd_graph, 
    device="cuda", 
    retriever=retriever,
    use_context = True, 
    top_k=5,
    hops=2, 
    graph_based=True
)

head, tail = [], []
for index, idn in idx2idn.items():
    neighbors = gnd_graph.neighbors(idn)
    neighbors_idx = [idn2idx[n_idn] for n_idn in neighbors]
    for n_idx in neighbors_idx:
        head.append(index)
        tail.append(n_idx)

edge_index = torch.tensor([head, tail])
x = torch.tensor(list(idx2idn.keys()))
data = Data(x=x, edge_index=edge_index)

collator.add_graph_data(
    idx2idn=idx2idn,
    idn2idx=idn2idx, 
    pyg_data=data
)
sample_docs = ds["train"].select(range(2))

gnn_pt = GraphContextPromptGenerator(config=default_config["prompt_config"], embeddings=label_embeddings)
loader = DataLoader(
    sample_docs,
    batch_size=1, 
    shuffle=False,
    collate_fn=collator 
)

for batch in loader:
    print(batch)
    break

# def get_graph_data(graph, embeddings=None, idx2idn=None):
#     if embeddings is None or idx2idn is None:
#         idx2idn, embeddings = self.retriever.embeddings()
#     embeddings = torch.tensor(embeddings)
#     idn2idx = {idn: idx for idx, idn in idx2idn.items()}
#     head, tail = [], []
#     for index, idn in idx2idn.items():
#         neighbors = graph.neighbors(idn)
#         neighbors_idx = [idn2idx[n_idn] for n_idn in neighbors]
#         for n_idx in neighbors_idx:
#             head.append(index)
#             tail.append(n_idx)
#     graph_data = {
#         "data": pyg.data.Data(x=embeddings, edge_index=torch.tensor([head, tail])),
#         "idn2idx": idn2idx, 
#         "idx2idn": idx2idn
#         }
#     return graph_data

