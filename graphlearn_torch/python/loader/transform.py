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

from typing import Dict, Optional

import torch
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

  if sampler_out.metadata is None:
    data.batch = sampler_out.batch
    data.batch_size = sampler_out.batch.numel() if data.batch is not None else 0
  elif isinstance(sampler_out.metadata, dict):
    if 'edge_label_index' in sampler_out.metadata:
      # binary negative sampling
      # In this case, we reverse the edge_label_index and put it into the
      # reversed edgetype subgraph
      edge_label_index = torch.stack(
        (sampler_out.metadata['edge_label_index'][1],
          sampler_out.metadata['edge_label_index'][0]), dim=0)
      data.edge_label_index = edge_label_index
      data.edge_label = sampler_out.metadata['edge_label']
    elif 'src_index' in sampler_out.metadata:
      # triplet negative sampling
      # In this case, src_index and dst_pos/neg_index fields follow the nodetype
      data.src_index = sampler_out.metadata['src_index']
      data.dst_pos_index = sampler_out.metadata['dst_pos_index']
      data.dst_neg_index = sampler_out.metadata['dst_neg_index']
    else:
      pass

  return data


def to_hetero_data(
  hetero_sampler_out: HeteroSamplerOutput,
  batch_label_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
  node_feat_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
  edge_feat_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
  **kwargs
) -> HeteroData:
  data = HeteroData(**kwargs)
  edge_index_dict = hetero_sampler_out.get_edge_index()
  # edges
  for k, v in edge_index_dict.items():
    data[k].edge_index = v
    if hetero_sampler_out.edge is not None:
      data[k].edge = hetero_sampler_out.edge.get(k, None)
    if edge_feat_dict is not None:
      data[k].edge_attr = edge_feat_dict.get(k, None)
  # nodes
  for k, v in hetero_sampler_out.node.items():
    data[k].node = v
    if node_feat_dict is not None:
      data[k].x = node_feat_dict.get(k, None)

  input_type = hetero_sampler_out.input_type
  # seed nodes
  if input_type is None or isinstance(input_type, NodeType):
    for k, v in hetero_sampler_out.batch.items():
      data[k].batch = v
      data[k].batch_size = v.numel()
      if batch_label_dict is not None:
        data[k].y = batch_label_dict.get(k, None)
  # seed edges
  else:
    rev_input_type = reverse_edge_type(input_type)
    if 'edge_label_index' in hetero_sampler_out.metadata:
      # binary negative sampling
      # In this case, we reverse the edge_label_index and put it into the
      # reversed edgetype subgraph
      edge_label_index = torch.stack(
        (hetero_sampler_out.metadata['edge_label_index'][1],
         hetero_sampler_out.metadata['edge_label_index'][0]), dim=0)
      data[rev_input_type].edge_label_index = edge_label_index
      data[rev_input_type].edge_label = hetero_sampler_out.metadata['edge_label']
    elif 'src_index' in hetero_sampler_out.metadata:
      # triplet negative sampling
      # In this case, src_index and dst_pos/neg_index fields follow the nodetype
      data[rev_input_type[-1]].src_index = hetero_sampler_out.metadata['src_index']
      data[rev_input_type[0]].dst_pos_index = hetero_sampler_out.metadata['dst_pos_index']
      data[rev_input_type[0]].dst_neg_index = hetero_sampler_out.metadata['dst_neg_index']

  return data
