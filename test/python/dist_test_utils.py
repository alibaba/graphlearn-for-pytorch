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

import torch
import graphlearn_torch as glt
from typing import Literal


# options for dataset generation
vnum_per_partition = 20
vnum_total = vnum_per_partition * 2
degree = 2
enum_total = vnum_total * degree

# for hetero dataset
user_ntype = 'user'
item_ntype = 'item'
u2i_etype = ('user', 'u2i', 'item')
i2i_etype = ('item', 'i2i', 'item')
rev_u2i_etype = ('item', 'rev_u2i', 'user')

# fixed sampling options
sampling_nprocs = 2
device_num = 2


def _prepare_dataset(rank: int, weighted: bool = False):
  # partition
  node_pb = torch.tensor(
    [v % 2 for v in range(0, vnum_total)],
    dtype=torch.long
  )
  edge_pb = torch.tensor(
    [((e // degree) % 2) for e in range(0, enum_total)],
    dtype=torch.long
  )

  # graph
  nodes, rows, cols, eids = [], [], [], []
  for v in range(rank, vnum_total, 2):
    nodes.append(v)
    rows.extend([v for _ in range(degree)])
    cols.extend([((v + i + 1) % vnum_total) for i in range(degree)])
    eids.extend([(v * degree + i) for i in range(degree)])
  edge_index = torch.tensor([rows, cols], dtype=torch.int64)
  edge_ids = torch.tensor(eids, dtype=torch.int64)
  edge_weights = (edge_ids % 2).to(torch.float)
  csr_topo = glt.data.Topology(edge_index=edge_index, edge_ids=edge_ids)
  weighted_csr_topo = glt.data.Topology(
    edge_index=edge_index, edge_ids=edge_ids, edge_weights=edge_weights)
  graph = glt.data.Graph(csr_topo, 'ZERO_COPY', device=0)
  weighted_graph = glt.data.Graph(weighted_csr_topo, 'CPU')

  # feature
  device_group_list = [glt.data.DeviceGroup(0, [0]),
                       glt.data.DeviceGroup(1, [1])]
  split_ratio = 0.2

  nfeat = rank + torch.zeros(len(nodes), 512, dtype=torch.float32)
  nfeat_id2idx = glt.utils.id2idx(nodes)
  node_feature = glt.data.Feature(nfeat, nfeat_id2idx, split_ratio,
                                  device_group_list, device=0)

  efeat = rank + torch.ones(len(eids), 10, dtype=torch.float32)
  efeat_id2idx = glt.utils.id2idx(eids)
  edge_feature = glt.data.Feature(efeat, efeat_id2idx, split_ratio,
                                  device_group_list, device=0)

  # whole node label
  node_label = torch.arange(vnum_total)

  # dist dataset
  if weighted:
    return glt.distributed.DistDataset(
      2, rank,
      weighted_graph, node_feature, edge_feature, node_label,
      node_pb, edge_pb
    )
  else:
    return glt.distributed.DistDataset(
      2, rank,
      graph, node_feature, edge_feature, node_label,
      node_pb, edge_pb
    )


def _prepare_hetero_dataset(
  rank: int,
  edge_dir: Literal['in', 'out'] = 'out',
  weighted: bool = False
):
  # partition
  node_pb = torch.tensor(
    [v % 2 for v in range(0, vnum_total)],
    dtype=torch.long
  )
  edge_pb = torch.tensor(
    [((e // degree) % 2) for e in range(0, enum_total)],
    dtype=torch.long
  )
  node_pb_dict = {
    user_ntype: node_pb,
    item_ntype: node_pb
  }
  edge_pb_dict = {
    u2i_etype: edge_pb,
    i2i_etype: edge_pb
  }

  # graph
  user_nodes = []
  u2i_rows = []
  u2i_cols = []
  u2i_eids = []
  for v in range(rank, vnum_total, 2):
    user_nodes.append(v)
    u2i_rows.extend([v for _ in range(degree)])
    u2i_cols.extend([((v + i + 1) % vnum_total) for i in range(degree)])
    u2i_eids.extend([(v * degree + i) for i in range(degree)])
  u2i_edge_index = torch.tensor([u2i_rows, u2i_cols], dtype=torch.int64)
  u2i_edge_ids = torch.tensor(u2i_eids, dtype=torch.int64)
  u2i_edge_weights = (u2i_edge_ids % 2).to(torch.float)
  if edge_dir == 'out':
    u2i_topo = glt.data.Topology(
      edge_index=u2i_edge_index, edge_ids=u2i_edge_ids, layout='CSR')
    weighted_u2i_topo = glt.data.Topology(
      edge_index=u2i_edge_index, edge_ids=u2i_edge_ids,
      edge_weights=u2i_edge_weights, layout='CSR')
  elif edge_dir == 'in':
    u2i_topo = glt.data.Topology(
      edge_index=u2i_edge_index, edge_ids=u2i_edge_ids, layout='CSC')
    weighted_u2i_topo = glt.data.Topology(
      edge_index=u2i_edge_index, edge_ids=u2i_edge_ids,
      edge_weights=u2i_edge_weights, layout='CSC')
  u2i_graph = glt.data.Graph(u2i_topo, 'ZERO_COPY', device=0)
  weighted_u2i_graph = glt.data.Graph(weighted_u2i_topo, 'CPU')

  item_nodes = []
  i2i_rows = []
  i2i_cols = []
  i2i_eids = []
  for v in range(rank, vnum_total, 2):
    item_nodes.append(v)
    i2i_rows.extend([v for _ in range(degree)])
    i2i_cols.extend([((v + i + 2) % vnum_total) for i in range(degree)])
    i2i_eids.extend([(v * degree + i) for i in range(degree)])
  i2i_edge_index = torch.tensor([i2i_rows, i2i_cols], dtype=torch.int64)
  i2i_edge_ids = torch.tensor(i2i_eids, dtype=torch.int64)
  i2i_edge_weights = (i2i_edge_ids % 2).to(torch.float)
  if edge_dir == 'out':
    i2i_topo = glt.data.Topology(
      edge_index=i2i_edge_index, edge_ids=i2i_edge_ids, layout='CSR')
    weighted_i2i_topo = glt.data.Topology(
      edge_index=i2i_edge_index, edge_ids=i2i_edge_ids,
      edge_weights=i2i_edge_weights, layout='CSR')
  elif edge_dir == 'in':
    i2i_topo = glt.data.Topology(
      edge_index=i2i_edge_index, edge_ids=i2i_edge_ids, layout='CSC')
    weighted_i2i_topo = glt.data.Topology(
      edge_index=i2i_edge_index, edge_ids=i2i_edge_ids,
      edge_weights=i2i_edge_weights, layout='CSC')
  i2i_graph = glt.data.Graph(i2i_topo, 'ZERO_COPY', device=0)
  weighted_i2i_graph = glt.data.Graph(weighted_i2i_topo, 'CPU')

  graph_dict = {
    u2i_etype: u2i_graph,
    i2i_etype: i2i_graph
  }

  weighted_graph_dict = {
    u2i_etype: weighted_u2i_graph,
    i2i_etype: weighted_i2i_graph
  }

  # feature
  device_group_list = [glt.data.DeviceGroup(0, [0]),
                       glt.data.DeviceGroup(1, [1])]
  split_ratio = 0.2

  user_nfeat = rank + torch.zeros(len(user_nodes), 512, dtype=torch.float32)
  user_nfeat_id2idx = glt.utils.id2idx(user_nodes)
  user_feature = glt.data.Feature(user_nfeat, user_nfeat_id2idx,
                                  split_ratio, device_group_list, device=0)

  item_nfeat = rank * 2 + torch.zeros(len(item_nodes), 256, dtype=torch.float32)
  item_nfeat_id2idx = glt.utils.id2idx(item_nodes)
  item_feature = glt.data.Feature(item_nfeat, item_nfeat_id2idx,
                                  split_ratio, device_group_list, device=0)

  node_feature_dict = {
    user_ntype: user_feature,
    item_ntype: item_feature
  }

  u2i_efeat = rank + torch.ones(len(u2i_eids), 10, dtype=torch.float32)
  u2i_efeat_id2idx = glt.utils.id2idx(u2i_eids)
  u2i_feature = glt.data.Feature(u2i_efeat, u2i_efeat_id2idx,
                                 split_ratio, device_group_list, device=0)

  i2i_efeat = rank * 2 + torch.ones(len(i2i_eids), 5, dtype=torch.float32)
  i2i_efeat_id2idx = glt.utils.id2idx(i2i_eids)
  i2i_feature = glt.data.Feature(i2i_efeat, i2i_efeat_id2idx,
                                 split_ratio, device_group_list, device=0)

  edge_feature_dict = {
    u2i_etype: u2i_feature,
    i2i_etype: i2i_feature
  }

  # node label
  node_label_dict = {
    user_ntype: torch.arange(vnum_total),
    item_ntype: torch.arange(vnum_total)
  }

  # dist dataset
  if weighted:
    return glt.distributed.DistDataset(
      2, rank,
      weighted_graph_dict, node_feature_dict, edge_feature_dict, node_label_dict,
      node_pb_dict, edge_pb_dict, edge_dir=edge_dir
    )
  else:
    return glt.distributed.DistDataset(
      2, rank,
      graph_dict, node_feature_dict, edge_feature_dict, node_label_dict,
      node_pb_dict, edge_pb_dict, edge_dir=edge_dir
    )
