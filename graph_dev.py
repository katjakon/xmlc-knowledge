import pickle
from gnd_graph import GNDGraph
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
#title_embeddings = retriever.retriever.encode(title_strings, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
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

    def forward(self, data: pyg.data.HeteroData) -> Tensor:
        x_dict = {
          "label": self.label_embed(data["label"].node_id),
          "title": data["title"].x,
        } 
        x = self.hetero_conv(x_dict, data.edge_index_dict)
        x = {k: F.relu(v) for k, v in x.items()}
        return x

gnn = GNN(dim)

print(data)

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
    neg_sampling_ratio=0.0,
    edge_label_index=(("title", "associate", "label"), edge_label_index),
    edge_label=edge_label,
    batch_size=1,
    shuffle=True,
)

for batch in train_loader:
    print(batch)
    out = gnn(batch)
    print(out)
    break


# graph_data = {
# "data": pyg.data.Data(x=embeddings, edge_index=torch.tensor([head, tail])),
# "idn2idx": idn2idx, 
# "idx2idn": idx2idn
# }

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
# collator = DataCollator(
#     tokenizer=tokenizer,
#     graph=gnd, 
#     device=device,
#     retriever=retriever,
#     use_context=True,
#     graph_based=True,
#     top_k=5, 
#     hops=2
# )
# graph_data = collator.get_graph_data()



# dataloader = DataLoader(
#     ds["test"], 
#     batch_size=8, 
#     collate_fn=collator)
# gat = pyg.nn.GAT(
#             in_channels=dim,
#             out_channels=dim,
#             hidden_channels=256,
#             num_layers=2)

# config = {
#     "hidden_size": 1024,
#     "num_prompt_tokens": 20,
#     "down_project_size": 512, 
#     "kge_size": dim,
#     "gnn_hidden_size": 256,
#     "gnn_n_layers": 2,
#     "dropout": 0.1
# }
# gnn_pg = GraphContextPromptGenerator(config)

# c = 0
# for i in dataloader:
#     c += 1
#     if c >= 2:
#         break
#     g_batch = i["graph_batch"]
#     size = i["input_ids"].size()
#     hidden_states = torch.rand((size[0], size[1], config["hidden_size"]))
#     out =  gnn_pg(
#         graph_batch=g_batch, 
#         hidden_states=hidden_states, 
#         seq_lengths=i["seq_lengths"]
#     )
#     print(out)
#     print(out.shape)
