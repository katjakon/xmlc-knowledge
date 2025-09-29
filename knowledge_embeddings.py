from torch_geometric.nn import TransE
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import TransE, GAT
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, to_hetero, GATConv
import torch.nn.functional as F
import torch
from torch import Tensor
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
data = Data(x=label_embeddings, edge_index=edge_index)
# data = HeteroData()
# data["label"].node_id  = torch.arange(len(mapping))
# data["label"].x = label_embeddings
# data["label", "connected", "label"].edge_index  = edge_index
# data = ToUndirected()(data)

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
)
#     rev_edge_types=("label", "rev_connected", "label")
train_data, val_data, test_data = transform(data)

edge_label_index = train_data.edge_label_index
# edge_label = train_data.edge_label
# edge_label_index = train_data["label", "connected", "label"].edge_label_index
# edge_label = train_data["label", "connected", "label"].edge_label
print(edge_label_index)
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=edge_label_index,
    # edge_label=edge_label,
    batch_size=128,
    shuffle=True,
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
    def forward(self, x_head: Tensor, x_tail: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_head = x_head[edge_label_index[0]]
        edge_feat_tail = x_tail[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_head * edge_feat_tail).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, x_size):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.label_lin = torch.nn.Linear(x_size, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        #self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: Data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x, edge_index)
        pred = self.classifier(
            x,
            x,
            edge_index,
        )
        return pred

model = Model(hidden_channels=128, x_size=384).to("cuda")

for i in train_loader:
    print(i)
    i.to("cuda")
    out = model(i)
    print(out)
    break

