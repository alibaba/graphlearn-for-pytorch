# Heterogeneous GNN

Many real-world graphs are heterogeneous, and the heterogeneous GNN algorithm is effective at the task of processing such graphs. We describe the GNN training process on heterogeneous graphs through a node classification example on the
[OGBN-MAG dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag).
The example is modified from PyG's [OGBN-MAG example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py)

## Loading and preprocessing OGBN-MAG dataset.

The OGBN-MAG dataset is a heterogeneous graph which contains four types of entities
and four types of directed relations.
We perform the same preprocessing as PyG: transforming the graph to an undirected graph and using MetaPath2Vec to obtain the node features.

``` python
import os.path as osp

import torch
import torch.nn.functional as F
import graphlearn_torch as glt
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import Linear, HGTConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/')
transform = T.ToUndirected(merge=True)
dataset = OGB_MAG(path, preprocess='metapath2vec', transform=transform)
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Creating data loader.
GLT uses a Python `Dict` instance to represent heterogeneous graphs, each element
of the dict represents a homogeneous graph.
In details, the graph topology information is stored through the dictionary of edge indices,
and the features of different nodes are stored through the dictionary of node features.

We create a dataset and initialize it with `edge_dict` and node `feature_dict`.
The graph data is stored in pinned memory, since the `graph_mode` is set to `ZERO_COPY`.
The node features are split into two parts according to `split_ratio`.
The first 20% feature data is stored in GPU 0 memory and the remaining 80%
is in pinned memory for ZERO-COPY access.

Then we define the training loader and the test loader;
they take paper nodes as input seed nodes and sample the 2-hop neighbors ([10] * 2) for each input node.
Similar to PyG's heterogeneous graph sampling, each hop sampling will be performed
on all types of edges that have a connection to the current node.

```python
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
```

## Defining model.

We directly use PyG's HGT implementation.
```python
class HGT(torch.nn.Module):
  def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
    super().__init__()

    self.lin_dict = torch.nn.ModuleDict()
    for node_type in data.node_types:
      self.lin_dict[node_type] = Linear(-1, hidden_channels)

    self.convs = torch.nn.ModuleList()
    for _ in range(num_layers):
      conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                     num_heads, group='sum')
      self.convs.append(conv)
    self.lin = Linear(hidden_channels, out_channels)

  def forward(self, x_dict, edge_index_dict):
    x_dict = {
      node_type: self.lin_dict[node_type](x).relu_()
      for node_type, x in x_dict.items()
    }
    for conv in self.convs:
      x_dict = conv(x_dict, edge_index_dict)
    return self.lin(x_dict['paper'])

# model
model = HGT(hidden_channels=64,
            out_channels=dataset.num_classes,
            num_heads=2,
            num_layers=2).to(device)
```

## Training and testing.

Finally, train the model and evaluate the model accuracy as follows:

``` python
@torch.no_grad()
def init_params():
  # Initialize lazy parameters via forwarding a single batch to the model:
  batch = next(iter(train_loader))
  model(batch.x_dict, batch.edge_index_dict)

def train():
  model.train()
  total_examples = total_loss = 0
  for batch in tqdm(train_loader):
    optimizer.zero_grad()
    batch_size = batch['paper'].batch_size
    out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
    loss = F.cross_entropy(out, batch['paper'].y[:batch_size])
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
    batch_size = batch['paper'].batch_size
    out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
    pred = out.argmax(dim=-1)
    total_examples += batch_size
    total_correct += int((pred == batch['paper'].y[:batch_size]).sum())
  return total_correct / total_examples

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(1, 21):
  loss = train()
  val_acc = test(val_loader)
  print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
```

This example is training on single GPU; for the multi-GPU training version,
please refer to the `examples/hetero/train_hgt_mag_mp.py`.