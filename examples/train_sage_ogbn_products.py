# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Reaches around 0.7870 ± 0.0036 test accuracy.
import time
import torch

import graphlearn_torch as glt
import os.path as osp
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


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

root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]


# PyG NeighborSampler
test_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                              batch_size=4096, shuffle=False, num_workers=12)

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

device = torch.device(0)
# graphlearn_torch NeighborLoader
train_loader = glt.loader.NeighborLoader(glt_dataset,
                                         [15, 10, 5],
                                         split_idx['train'],
                                         batch_size=1024,
                                         shuffle=True,
                                         drop_last=True,
                                         device=device,
                                         as_pyg_v1=True)

model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

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
