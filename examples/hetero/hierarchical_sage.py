import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from torch_geometric.utils import trim_to_layer
import graphlearn_torch as glt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transforms = [T.ToUndirected(merge=True)]
dataset = OGB_MAG(root='../../data', preprocess='metapath2vec',
                  transform=T.Compose(transforms))
data = dataset[0].to(device, 'x', 'y')


class HierarchicalHeteroGraphSage(torch.nn.Module):
    def __init__(self, edge_types, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((-1, -1), hidden_channels)
                    for edge_type in edge_types
                }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_sampled_edges_dict,
                num_sampled_nodes_dict):
        
        for i, conv in enumerate(self.convs):
            x_dict, edge_index_dict, _ = trim_to_layer(
                layer=i,
                num_sampled_nodes_per_hop=num_sampled_nodes_dict,
                num_sampled_edges_per_hop=num_sampled_edges_dict,
                x=x_dict,
                edge_index=edge_index_dict,
            )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['paper'])


model = HierarchicalHeteroGraphSage(
    edge_types=data.edge_types,
    hidden_channels=64,
    out_channels=dataset.num_classes,
    num_layers=2,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  # init graphlearn_torch Dataset.
edge_dict, feature_dict = {}, {}
for etype in data.edge_types:
  edge_dict[etype] = data[etype]['edge_index']
for ntype in data.node_types:
  feature_dict[ntype] = data[ntype].x.clone(memory_format=torch.contiguous_format)

glt_dataset = glt.data.Dataset()
glt_dataset.init_graph(
  edge_index=edge_dict,
  graph_mode='ZERO_COPY'
)
glt_dataset.init_node_features(
  node_feature_data=feature_dict,
  split_ratio=0.2,
  device_group_list=[glt.data.DeviceGroup(0, [0])]
)
glt_dataset.init_node_labels(node_label_data={'paper': data['paper'].y})

train_idx = data['paper'].train_mask.nonzero(as_tuple=False).view(-1)
train_loader = glt.loader.NeighborLoader(glt_dataset,
                                        [10] * 2,
                                        ('paper', train_idx),
                                        batch_size=1024,
                                        shuffle=True,
                                        device=device)
val_idx = data['paper'].val_mask.nonzero(as_tuple=False).view(-1)
val_loader = glt.loader.NeighborLoader(glt_dataset,
                                        [10] * 2,
                                        ('paper', val_idx),
                                        batch_size=1024,
                                        device=device)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            num_sampled_nodes_dict=batch.num_sampled_nodes,
            num_sampled_edges_dict=batch.num_sampled_edges,
        )

        batch_size = batch['paper'].batch_size
        loss = F.cross_entropy(out[:batch_size], batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            num_sampled_nodes_dict=batch.num_sampled_nodes,
            num_sampled_edges_dict=batch.num_sampled_edges,
        )

        batch_size = batch['paper'].batch_size
        pred = out[:batch_size].argmax(dim=-1)
        total_examples += batch_size
        total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

    return total_correct / total_examples


for epoch in range(1, 6):
    loss = train()
    val_acc = test(val_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
