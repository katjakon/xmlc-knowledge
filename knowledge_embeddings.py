import argparse
import pickle
import os
import json

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

from gnd_graph import GNDGraph
from gnd_dataset import GNDDataset

device = "cuda"
gnd_path  = "gnd/gnd.pickle"
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

data_path = "dataset"

label_strings, mapping = gnd_graph.mapping()
idn2idx = {value: key for key, value in mapping.items()}

model = SentenceTransformer(
    'BAAI/bge-m3',
    device=device
    )
dim = model.get_sentence_embedding_dimension()

data_path = "dataset"
ds = GNDDataset(
    data_dir=data_path,
    gnd_graph=gnd_graph, 
    load_from_disk=True,
)
subsample = True
n_epochs = 20
lr = 0.001
warmup_rate = 0.03
out_dir = "kge"

head, tail = [], []

for index, idn in tqdm(mapping.items()):
    neighbors = gnd_graph.neighbors(idn)
    neighbors_idx = [idn2idx[n_idn] for n_idn in neighbors]
    for n_idx in neighbors_idx:
        head.append(index)
        tail.append(n_idx)

label_embeddings = model.encode(
    label_strings, 
    batch_size=1024,
    show_progress_bar=True,
    convert_to_tensor=True
    )

train_docs = ds["train"]
if subsample:
    train_docs = train_docs.select(range(100))
title_strings = list(train_docs["title"])
title_embeddings = model.encode(
    title_strings, 
    batch_size=1024, 
    show_progress_bar=True,
    convert_to_tensor=True
    )

embeddings = torch.cat([label_embeddings, title_embeddings])
for doc_record in tqdm(train_docs):
    index += 1
    title = doc_record["title"]
    gold_labels = doc_record["label-ids"]
    idn = doc_record["doc_idn"]
    for label_idn in gold_labels:
        label_idx = idn2idx.get(label_idn)
        if label_idx is None:
            continue
        head.append(index)
        tail.append(label_idx)

edge_index = torch.tensor([head, tail], dtype=torch.long)
data = Data(x=embeddings, edge_index=edge_index)

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.0,
    is_undirected=False,
    disjoint_train_ratio=0.1,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
)

train_data, val_data, test_data = transform(data)

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=train_data.edge_index,
    batch_size=128,
    shuffle=True,
)
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=val_data.edge_index,
    batch_size=3 * 128,
    shuffle=False,
)

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_feats: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_head = x_feats[edge_label_index[0]]
        edge_feat_tail = x_feats[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        dot_product = edge_feat_head * edge_feat_tail
        result = dot_product.sum(dim=-1)
        return result

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, x_size):
        super().__init__()
        self.label_lin = torch.nn.Linear(x_size, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        self.classifier = Classifier()

    def forward(self, data: Data) -> Tensor:
        # Edge index is used for message passing. Includes neighbors etc.
        # Edge label index holds sampled edges which we want to predict.
        edge_label_index = None
        x, edge_index = data.x,data.edge_index
        if "edge_label_index" in data:
            edge_label_index = data.edge_label_index 
        else:
            edge_label_index = edge_index
        x = self.label_lin(x)
        x = self.gnn(x, edge_index)
        pred = self.classifier(x, edge_label_index)
        return pred, x

total_steps = len(train_loader) * n_epochs
model = Model(hidden_channels=128, x_size=dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(warmup_rate* total_steps),
    num_training_steps=total_steps,
)

for epoch in range(n_epochs):
    total_loss = total_examples = 0
    for sampled_data in tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        ground_truth = sampled_data.edge_label
        pred, _ = model(sampled_data)
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
        lr = scheduler.get_last_lr()[0]
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}. LR: {lr:.4f}")

    preds = []
    ground_truths = []
    for sampled_data in tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            pred, _ = model(sampled_data)
            pred = F.sigmoid(pred)
            preds.append(pred)
            ground_truths.append(sampled_data.edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    pred_binary = ( pred >= 0.5).astype(int)     
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy().astype(int)  
    auc = roc_auc_score(ground_truth, pred)
    prec = precision_recall_fscore_support(ground_truth, pred_binary, average="binary")
    print()
    print(f"Validation AUC: {auc:.4f}")
    print(f"Precision: {prec[0]:.4f}")
    print(f"Recall: {prec[1]:.4f}")
    print(f"F1: {prec[2]:.4f}")

torch.save(model.state_dict(), os.path.join(out_dir, "kge_model.pt"))
# Generate embeddings for all labels
with torch.no_grad():
    data.to(device)
    pred, x = model(data)
torch.save(x, os.path.join(out_dir, "embeddings.pt"))
mapping_path = os.path.join(out_dir, "mapping.json")
with open(mapping_path, 'w') as f:
    json.dump(mapping, f, indent=4)