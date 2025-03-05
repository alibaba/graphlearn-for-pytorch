# Node Classification

Node classification is the basic task of graph learning and GNNs are powerful
models of graph learning.
We introduce the basic workflow of GNN training through the node classification
example on [OGBN-Products dataset](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products).
The code is based on PyG's signle GPU training of
[GraphSAGE on OGBN-Products](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py) and the only difference is to use GLT's
[`graphlearn_torch.loader.neighbor_loader.NeighborLoader`](graphlearn_torch.loader.neighbor_loader.NeighborLoader)
instead of PyG's `torch_geometric.loader.NeighborSampler` to accelerate training on GPU.
For model testing, we keep the original NeighborSampler in PyG to do the usage comparison.

## Loading OGBN-Products dataset.

``` python
import time
import torch

import graphlearn_torch as glt
import os.path as osp
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]

# PyG NeighborSampler
test_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                              batch_size=4096, shuffle=False, num_workers=12)
```

> **Note**
> In PyG 1.x, NeighborSampler is actually neighbor loader.

## Creating data loader.

This part is the only difference from PyG's example.
We first create an instance of [`graphlearn_torch.data.dataset.Dataset`](graphlearn_torch.data.dataset.Dataset)
and initialize it with edge_index and node features.

The graph data is stored in pinned memory, since the `graph_mode` is set to `ZERO_COPY`.
The `graph_mode` can be `GPU`, `ZERO_COPY` or `CPU` indicating the data is
stored in GPU memory, pinned memory and CPU memory, respectively. `GPU` and `ZERO_COPY`
are recommended, and you should choose `ZERO_COPY` when graph data is larger than
GPU memory capacity.
The node features are sorted by in-degrees of nodes and then split into two parts
according to `split_ratio`. The node features with higher in-degrees are stored
in GPU memory, and the remaining part is stored in pinned memory for UVA.

Then, we use a neigbor loader [`graphlearn_torch.loader.neighbor_loader.NeighborLoader`](graphlearn_torch.loader.neighbor_loader.NeighborLoader) which is totally compatible with PyG's `NeighborSampler`.
This loader uses train index as input seeds and samples 3-hop of neighbors for each
seed.
In the overall process of large-scale graph GNN GPU training, sampling and feature
lookup often become bottlenecks due to the low bandwidth of the PCIe and the
limited concurrency of the CPU.
In GLT, sampling and feature lookup are executed on the GPU, which provides a significant
performance boost compared to the CPU.

``` python
glt_dataset = glt.data.Dataset()
glt_dataset.init_graph(
  edge_index=dataset[0].edge_index,
  graph_mode='ZERO_COPY',
  directed=False
)
glt_dataset.init_node_features(
  node_feature_data=data.x,
  sort_func=glt.data.sort_by_in_degree,
  split_ratio=0.2,
  device_group_list=[glt.data.DeviceGroup(0, [0])],
)
glt_dataset.init_node_labels(node_label_data=data.y)

# graphlearn_torch NeighborLoader
train_loader = glt.loader.NeighborLoader(glt_dataset,
                                         [15, 10, 5],
                                         split_idx['train'],
                                         batch_size=1024,
                                         shuffle=True,
                                         drop_last=True,
                                         device=torch.device('cuda:0'),
                                         as_pyg_v1=True)
```
> **Note**
> In PyG 2.x, the neighbor sampler output has been changed from that in PyG 1.x,
> so we add the argument `as_pyg_v1` to support sampling in PyG 1.x.

## Defining model.

Here we directly show the PyG's GraphSAGE model defination.
```python
class SAGE(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
    super(SAGE, self).__init__()

    self.num_layers = num_layers

    self.convs = torch.nn.ModuleList()
    self.convs.append(SAGEConv(in_channels, hidden_channels))
    for _ in range(num_layers - 2):
      self.convs.append(SAGEConv(hidden_channels, hidden_channels))
    self.convs.append(SAGEConv(hidden_channels, out_channels))

  def reset_parameters(self):
    for conv in self.convs:
      conv.reset_parameters()

  def forward(self, x, adjs):
    # `train_loader` computes the k-hop neighborhood of a batch of nodes,
    # and returns, for each layer, a bipartite graph object, holding the
    # bipartite edges `edge_index`, the index `e_id` of the original edges,
    # and the size/shape `size` of the bipartite graph.
    # Target nodes are also included in the source nodes so that one can
    # easily apply skip-connections or add self-loops.
    for i, (edge_index, _, size) in enumerate(adjs):
      x_target = x[:size[1]]  # Target nodes are always placed first.
      x = self.convs[i]((x, x_target), edge_index)
      if i != self.num_layers - 1:
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
    return x.log_softmax(dim=-1)

  def inference(self, x_all):
    pbar = tqdm(total=x_all.size(0) * self.num_layers)
    pbar.set_description('Evaluating')
    # Compute representations of nodes layer by layer, using *all*
    # available edges. This leads to faster computation in contrast to
    # immediately computing the final representations of each batch.
    total_edges = 0
    for i in range(self.num_layers):
      xs = []
      for batch_size, n_id, adj in test_loader:
        edge_index, _, size = adj.to(device)
        total_edges += edge_index.size(1)
        x = x_all[n_id].to(device)
        x_target = x[:size[1]]
        x = self.convs[i]((x, x_target), edge_index)
        if i != self.num_layers - 1:
          x = F.relu(x)
        xs.append(x.cpu())
        pbar.update(batch_size)
      x_all = torch.cat(xs, dim=0)
    pbar.close()
    return x_all

model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)
```

## Training and testing.

Finally, you can use the GLT's trainer_loader defined above to speed up your program.

``` python
def train(epoch):
  model.train()
  pbar = tqdm(total=split_idx['train'].size(0))
  pbar.set_description(f'Epoch {epoch:02d}')

  total_loss = total_correct = 0
  step = 0
  glt_dataset.node_labels = glt_dataset.node_labels.to(device)
  for batch_size, n_id, adjs in train_loader:
    # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
    adjs = [adj.to(device) for adj in adjs]
    optimizer.zero_grad()
    out = model(glt_dataset.node_features[n_id], adjs)
    loss = F.nll_loss(out, glt_dataset.node_labels[n_id[:batch_size]])
    loss.backward()
    optimizer.step()
    total_loss += float(loss)
    total_correct += int(out.argmax(dim=-1).eq(glt_dataset.node_labels[n_id[:batch_size]]).sum())
    step += 1
    pbar.update(batch_size)

  pbar.close()

  loss = total_loss / step
  approx_acc = total_correct / split_idx['train'].size(0)
  return loss, approx_acc


@torch.no_grad()
def test():
  model.eval()
  out = model.inference(glt_dataset.node_features)

  y_true = glt_dataset.node_labels.cpu().unsqueeze(-1)
  y_pred = out.argmax(dim=-1, keepdim=True)

  train_acc = evaluator.eval({
    'y_true': y_true[split_idx['train']],
    'y_pred': y_pred[split_idx['train']],
  })['acc']
  val_acc = evaluator.eval({
    'y_true': y_true[split_idx['valid']],
    'y_pred': y_pred[split_idx['valid']],
  })['acc']
  test_acc = evaluator.eval({
    'y_true': y_true[split_idx['test']],
    'y_pred': y_pred[split_idx['test']],
  })['acc']

  return train_acc, val_acc, test_acc


test_accs = []
for run in range(1, 2):
  print('')
  print(f'Run {run:02d}:')
  print('')

  model.reset_parameters()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

  best_val_acc = final_test_acc = 0
  for epoch in range(1, 21):
    epoch_start = time.time()
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}',
          f'Epoch Time: {time.time() - epoch_start}')

    if epoch > 5:
      train_acc, val_acc, test_acc = test()
      print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
      if val_acc > best_val_acc:
        best_val_acc = val_acc
        final_test_acc = test_acc
  test_accs.append(final_test_acc)

test_acc = torch.tensor(test_accs)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
```

This example can have about 1x performance improvement compared to the
original code, while increasing GPU utilization and decreasing CPU usage.