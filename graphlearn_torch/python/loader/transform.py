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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eithPer express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Dict, Optional, Literal

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from ..sampler import SamplerOutput, HeteroSamplerOutput
from ..typing import NodeType, EdgeType, reverse_edge_type


def to_data(
  sampler_out: SamplerOutput,
  batch_labels: Optional[torch.Tensor] = None,
  node_feats: Optional[torch.Tensor] = None,
  edge_feats: Optional[torch.Tensor] = None,
  **kwargs
) -> Data:
  edge_index = torch.stack([sampler_out.row, sampler_out.col])
  data = Data(x=node_feats, edge_index=edge_index,
              edge_attr=edge_feats, y=batch_labels, **kwargs)
  data.edge = sampler_out.edge
  data.node = sampler_out.node

  data.batch = sampler_out.batch
  data.batch_size = sampler_out.batch.numel() if data.batch is not None else 0

  data.num_sampled_nodes = sampler_out.num_sampled_nodes
  data.num_sampled_edges = sampler_out.num_sampled_edges

  # update meta data
  if isinstance(sampler_out.metadata, dict):
    for k, v in sampler_out.metadata.items():
      if k == 'edge_label_index':
        # In binary negative sampling from edges, we reverse the
        # edge_label_index and put it into the reversed edgetype subgraph.
        data['edge_label_index'] = torch.stack((v[1], v[0]), dim=0)
      else:
        data[k] = v
  elif sampler_out.metadata is not None:
    data['metadata'] = sampler_out.metadata

  return data


def to_hetero_data(
  hetero_sampler_out: HeteroSamplerOutput,
  batch_label_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
  node_feat_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
  edge_feat_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
  edge_dir: Literal['in', 'out'] = 'out',
  **kwargs
) -> HeteroData:
  data = HeteroData(**kwargs)
  edge_index_dict = hetero_sampler_out.get_edge_index()
  num_hops = max(map(
    lambda x: len(x), list(hetero_sampler_out.num_sampled_edges.values())))
  # edges
  for k, v in edge_index_dict.items():
    data[k].edge_index = v
    if hetero_sampler_out.edge is not None:
      data[k].edge = hetero_sampler_out.edge.get(k, None)
    if edge_feat_dict is not None:
      data[k].edge_attr = edge_feat_dict.get(k, None)
    if k not in hetero_sampler_out.num_sampled_edges:
      hetero_sampler_out.num_sampled_edges[k] = \
        torch.tensor([0] * num_hops, device=data[k].edge_index.device)
    else:
      hetero_sampler_out.num_sampled_edges[k] = F.pad(
        hetero_sampler_out.num_sampled_edges[k],
        (0, num_hops - hetero_sampler_out.num_sampled_edges[k].size(0))
      )

  # nodes
  for k, v in hetero_sampler_out.node.items():
    data[k].node = v
    if node_feat_dict is not None:
      data[k].x = node_feat_dict.get(k, None)
    if k not in hetero_sampler_out.num_sampled_nodes:
      hetero_sampler_out.num_sampled_nodes[k] = \
        torch.tensor([0] * (num_hops + 1), device=data[k].node.device)
    else:
      hetero_sampler_out.num_sampled_nodes[k] = F.pad(
        hetero_sampler_out.num_sampled_nodes[k],
        (0, num_hops + 1 - hetero_sampler_out.num_sampled_nodes[k].size(0))
      )

  # seed nodes
  for k, v in hetero_sampler_out.batch.items():
    data[k].batch = v
    data[k].batch_size = v.numel()
    if batch_label_dict is not None:
      data[k].y = batch_label_dict.get(k, None)

  # update num_sampled_nodes & num_sampled_edges
  data.num_sampled_nodes = hetero_sampler_out.num_sampled_nodes
  data.num_sampled_edges = hetero_sampler_out.num_sampled_edges

  # update meta data
  input_type = hetero_sampler_out.input_type
  if isinstance(hetero_sampler_out.metadata, dict):
    # if edge_dir == 'out', we need to reverse the edge type
    res_edge_type = reverse_edge_type(input_type) if edge_dir == 'out' else input_type
    for k, v in hetero_sampler_out.metadata.items():
      if k == 'edge_label_index':
        if edge_dir == 'out':
          data[res_edge_type]['edge_label_index'] = \
            torch.stack((v[1], v[0]), dim=0)
        else:
          data[res_edge_type]['edge_label_index'] = v
      elif k == 'edge_label':
        data[res_edge_type]['edge_label'] = v
      elif k == 'src_index':
        data[input_type[0]]['src_index'] = v
      elif k in ['dst_pos_index', 'dst_neg_index']:
        data[input_type[-1]][k] = v
      else:
        data[k] = v
  elif hetero_sampler_out.metadata is not None:
    data['metadata'] = hetero_sampler_out.metadata

  return data
