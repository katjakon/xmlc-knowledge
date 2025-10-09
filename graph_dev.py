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
from tqdm import tqdm

subsample = True
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

label_edge_index = [head, tail]
data["label"].node_id = torch.arange(len(idn2idx))
data["label"].x = embeddings
data["label", "connects", "label"].edge_index = label_edge_index

head_t, tail_t = [], []
train_docs = ds["train"].select(range(100))

all_title_embed = []
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
    title_embedding = retriever.retriever.encode([title])
    title_embedding = torch.tensor(title_embedding)
    all_title_embed.append(title_embedding)

print(head_t)
title_embeddings = torch.cat(all_title_embed)
data["title"].x = title_embeddings
data["title"].node_id = torch.arange(title_embeddings.size()[0])
data["title", "associate", "label"].edge_index = [head_t, tail_t]

print(data)

graph_data = {
"data": pyg.data.Data(x=embeddings, edge_index=torch.tensor([head, tail])),
"idn2idx": idn2idx, 
"idx2idn": idx2idn
}

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
