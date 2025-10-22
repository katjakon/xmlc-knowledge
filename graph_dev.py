import pickle
from gnd_graph import GNDGraph
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch
from torch import Tensor
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from retriever import Retriever
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
from data_collator import DataCollator
from gnd_dataset import GNDDataset
from prompt_generators import GraphContextPromptGenerator
from utils import PAD_TOKEN
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

subsample = False
device = "cuda" if torch.cuda.is_available() else "cpu"
gnd = pickle.load(open("gnd/gnd.pickle", "rb"))
gnd = GNDGraph(gnd)

if subsample:
    gnd = gnd.subgraph(list(gnd.nodes)[:20])

retriever = Retriever(retriever_model="distiluse-base-multilingual-cased-v1", graph=gnd) # 'BAAI/bge-m3'
retriever.fit()
dim = retriever.dim

data_path = "dataset"
ds = GNDDataset(
    data_dir=data_path,
    gnd_graph=gnd, 
    load_from_disk=True,
)

data = pyg.data.HeteroData()

idx2idn, embeddings = retriever.embeddings()
embeddings = torch.tensor(embeddings)
idn2idx = {idn: idx for idx, idn in idx2idn.items()}
head, tail = [], []
for index, idn in idx2idn.items():
    neighbors = gnd.neighbors(idn)
    neighbors_idx = [idn2idx[n_idn] for n_idn in neighbors]
    for n_idx in neighbors_idx:
        head.append(index)
        tail.append(n_idx)

label_edge_index =  torch.tensor([head, tail], dtype=torch.int64)
data["label"].node_id = torch.arange(len(idn2idx))
data["label"].x = embeddings
data["label", "connect", "label"].edge_index = label_edge_index
head_t, tail_t = [], []
train_docs = ds["train"]

if subsample:
    train_docs = train_docs.select(range(100))
title_strings = list(train_docs["title"])
# title_embeddings = retriever.retriever.encode(title_strings, batch_size=256, show_progress_bar=True)
# title_embeddings = torch.tensor(title_embeddings)
title_embeddings = torch.rand((len(title_strings), dim))
for index, doc_record in tqdm(enumerate(train_docs)):
    title = doc_record["title"]
    gold_labels = doc_record["label-ids"]
    idn = doc_record["doc_idn"]
    for label_idn in gold_labels:
        label_idx = idn2idx.get(label_idn)
        if label_idx is None:
            continue
        head_t.append(index)
        tail_t.append(label_idx)

title_edge_index = torch.tensor([head_t, tail_t], dtype=torch.int64)
data["title"].x = title_embeddings
data["title"].node_id = torch.arange(title_embeddings.size()[0])
data["title", "associate", "label"].edge_index = title_edge_index
data = T.ToUndirected()(data)

class Classifier(torch.nn.Module):
    def forward(self, x_title: Tensor, x_label:Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_head = x_title[edge_label_index[0]]
        edge_feat_tail = x_label[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_head * edge_feat_tail).sum(dim=-1)

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lbl_title_conv = pyg.nn.SAGEConv(hidden_channels, hidden_channels)
        self.lbl_lbl_conv = pyg.nn.SAGEConv(hidden_channels, hidden_channels)
        self.hetero_conv = pyg.nn.HeteroConv(
            {
            ("title", "associate", "label"): self.lbl_title_conv,
            ("label", "rev_associate", "title"): self.lbl_title_conv,
            ("label", "connect", "label"): self.lbl_lbl_conv
            },
         aggr='sum'
        )
        self.label_embed = torch.nn.Embedding(data["label"].num_nodes, hidden_channels)
        self.classifier = Classifier()

    def forward(self, data: pyg.data.HeteroData) -> Tensor:
        x_dict = {
          "label": self.label_embed(data["label"].node_id),
          "title": data["title"].x,
        } 
        x = self.hetero_conv(x_dict, data.edge_index_dict)
        x = {k: F.relu(v) for k, v in x.items()}
        x_title = x["title"]
        x_label = x["label"]
        title_edge_label_index = data[("title", "associate", "label")].edge_label_index
        pred = self.classifier( 
            x_title=x_title,
            x_label=x_label, 
            edge_label_index=title_edge_label_index)
        return pred, x

gnn = GNN(dim)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=False,
    disjoint_train_ratio=0.1,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("title", "associate", "label"),
    rev_edge_types=("label", "rev_associate", "title")
)

train_data, val_data, test_data = transform(data)

edge_label_index = train_data["title", "associate", "label"].edge_label_index
edge_label = train_data["title", "associate", "label"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("title", "associate", "label"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)
val_edge_label_index = val_data["title", "associate", "label"].edge_label_index
val_edge_label = val_data["title", "associate", "label"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("title", "associate", "label"), val_edge_label_index),
    edge_label=val_edge_label,
    batch_size=128*3,
    shuffle=True,
)

n_epochs = 20
lr = 0.001
warmup_rate = 0.03
total_steps = len(train_loader) * n_epochs
optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(warmup_rate* total_steps),
    num_training_steps=total_steps,
)
gnn.to(device)
for epoch in range(n_epochs):
    total_loss = total_examples = 0
    i = 0
    for batch in tqdm(train_loader):
        batch.to(device)
        i += 1
        optimizer.zero_grad()
        pred, x = gnn(batch)
        ground_truth = batch[("title", "associate", "label")].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
        lr = scheduler.get_last_lr()[0]
        if i % 50 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}. LR: {lr:.4f}")
        if i % 10 == 0:
            preds = []
            ground_truths = []
            for sampled_data in tqdm(val_loader):
                with torch.no_grad():
                    sampled_data.to(device)
                    pred, _ = gnn(sampled_data)
                preds.append(pred)
                ground_truths.append(sampled_data["title", "associate", "label"].edge_label)
            pred = torch.cat(preds, dim=0).cpu().numpy()
            pred_binary = ( pred >= 0.5).astype(int)     
            ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy().astype(int) 
            print(ground_truth.shape, pred.shape)  
            auc = roc_auc_score(ground_truth, pred)
            prec = precision_recall_fscore_support(ground_truth, pred_binary, average="binary")
            print()
            print(f"Validation AUC: {auc:.4f}")
            print(f"Precision: {prec[0]:.4f}")
            print(f"Recall: {prec[1]:.4f}")
            print(f"F1: {prec[2]:.4f}")
