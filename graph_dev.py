import pickle
from gnd_graph import GNDGraph
import torch
import torch_geometric as pyg
from retriever import Retriever
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data_collator import DataCollator
from gnd_dataset import GNDDataset
from utils import PAD_TOKEN

subsample = False
device = "cuda" if torch.cuda.is_available() else "cpu"
gnd = pickle.load(open("gnd/gnd.pickle", "rb"))
gnd = GNDGraph(gnd)

if subsample:
    gnd = gnd.subgraph(list(gnd.nodes)[:20])

retriever = Retriever(retriever_model='BAAI/bge-m3', graph=gnd)
retriever.fit()
dim = 1024

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
collator = DataCollator(
    tokenizer=tokenizer,
    graph=gnd, 
    device=device,
    retriever=retriever,
    use_context=True,
    graph_based=True,
    top_k=5, 
    hops=2
)
graph_data = collator.get_graph_data()

data_path = "dataset"
ds = GNDDataset(
    data_dir=data_path,
    gnd_graph=gnd, 
    load_from_disk=True,
)

dataloader = DataLoader(
    ds["test"], 
    batch_size=8, 
    collate_fn=collator)
gat = pyg.nn.GAT(
            in_channels=dim,
            out_channels=dim,
            hidden_channels=256,
            num_layers=2)
c = 0
for i in dataloader:
    c += 1
    if c >2:
        break
    g_batch = i["graph_batch"]
    sep_graphs = g_batch.to_data_list()
    print(sep_graphs)
    output = gat(
        x=g_batch.x,
        edge_index=g_batch.edge_index
    )
    _, counts = torch.unique(g_batch.batch, return_counts=True)
    original_batch = torch.split(output, counts.tolist())
    for graph in original_batch:
        print(graph.size())
