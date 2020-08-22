import torch
from torch_geometric.data import Data
from torch_scatter import scatter_mean

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

print(data.num_nodes)


from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset


dataset = TUDataset(root='data/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(type(dataset))

for batch in loader:
    # print(batch.batch)
    print(batch.num_graphs)
    # print(scatter_mean(batch.x, batch.batch, dim=0))


