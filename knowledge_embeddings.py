from torch_geometric.nn import TransE
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
import torch
import pickle
from gnd_graph import GNDGraph

gnd_path  = "gnd/gnd.pickle"
gnd_graph = pickle.load(open(gnd_path, "rb"))
gnd_graph = GNDGraph(gnd_graph)

label_strings, mapping = gnd_graph.mapping()
idn2idx = {value: key for key, value in mapping.items()}

model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    device="cuda"
    )

head, tail = [], []

for index, idn in mapping.items():
    neighbors = gnd_graph.neighbors(idn)
    neighbors_idx = [idn2idx[n_idn] for n_idn in neighbors]
    for n_idx in neighbors_idx:
        head.append(index)
        tail.append(n_idx)

label_embeddings = model.encode(
    label_strings, 
    show_progress_bar=True,
    convert_to_tensor=True
    )


edge_index = torch.tensor([head, tail], dtype=torch.long)
x = label_embeddings

data = Data(x=x, edge_index=edge_index)
print(data)

print(data.validate(raise_on_error=True))
print(data.num_node_features)
print(data.has_isolated_nodes())
print(data.has_self_loops())
print(data.is_directed())


# class SequenceEncoder:
#     def __init__(self, model_name="'all-MiniLM-L6-v2'", device=None):
#         self.device = device
#         self.model = SentenceTransformer(model_name, device=device)

#     @torch.no_grad()
#     def __call__(self, df):
#         x = self.model.encode(df.values, show_progress_bar=True,
#                               convert_to_tensor=True, device=self.device)
#         return x.cpu()