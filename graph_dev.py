import pickle
from gnd_graph import GNDGraph
import torch
import torch_geometric as pyg
from retriever import Retriever
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data_collator import DataCollator
from gnd_dataset import GNDDataset
from prompt_generators import GraphContextPromptGenerator
from utils import PAD_TOKEN

subsample = True
device = "cuda" if torch.cuda.is_available() else "cpu"
gnd = pickle.load(open("gnd/gnd.pickle", "rb"))
gnd = GNDGraph(gnd)

if subsample:
    gnd = gnd.subgraph(list(gnd.nodes)[:20])

retriever = Retriever(retriever_model="distiluse-base-multilingual-cased-v1", graph=gnd) # 'BAAI/bge-m3'
retriever.fit()
dim = retriever.dim

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

config = {
    "hidden_size": 1024,
    "num_prompt_tokens": 20,
    "down_project_size": 512, 
    "kge_size": dim,
    "gnn_hidden_size": 256,
    "gnn_n_layers": 2,
    "dropout": 0.1
}
gnn_pg = GraphContextPromptGenerator(config)

c = 0
for i in dataloader:
    c += 1
    if c >= 2:
        break
    g_batch = i["graph_batch"]
    size = i["input_ids"].size()
    hidden_states = torch.rand((size[0], size[1], config["hidden_size"]))
    out =  gnn_pg(
        graph_batch=g_batch, 
        hidden_states=hidden_states, 
        seq_lengths=i["seq_lengths"]
    )
    print(out)
    print(out.shape)
