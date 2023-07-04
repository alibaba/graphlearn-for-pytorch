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

from typing import Union

import torch

from ..data import Dataset
from ..sampler import BaseSampler, SamplerOutput, HeteroSamplerOutput
from ..typing import InputNodes

from .transform import to_data, to_hetero_data


class NodeLoader(object):
  r"""A base data loader that performs node sampling for mini-batch training
  of GNNs on large-scale graphs.

  Args:
    data (Dataset): The `graphlearn_torch.data.Dataset` object.
    node_sampler (graphlearn_torch.sampler.BaseSampler): The sampler
      implementation to be used with this loader.
      Needs to implement
      :meth:`~graphlearn_torch.sampler.BaseSampler.sample_from_nodes`.
      The sampler implementation must be compatible with the input
      :obj:`data` object.
    num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
      number of neighbors to sample for each node in each iteration.
      In heterogeneous graphs, may also take in a dictionary denoting
      the amount of neighbors to sample for each individual edge type.
      If an entry is set to :obj:`-1`, all neighbors will be included.
    input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
      indices of nodes for which neighbors are sampled to create
      mini-batches.
      Needs to be either given as a :obj:`torch.LongTensor` or
      :obj:`torch.BoolTensor`.
      In heterogeneous graphs, needs to be passed as a tuple that holds
      the node type and node indices.
    with_edge (bool): Set to ``True`` to sample with edge ids and also include
      them in the sampled results. (default: ``False``).
  """
  def __init__(
    self,
    data: Dataset,
    node_sampler: BaseSampler,
    input_nodes: InputNodes,
    device: torch.device = torch.device(0),
    **kwargs
  ):
    self.data = data
    self.sampler = node_sampler
    self.input_nodes = input_nodes
    self.device = device

    if isinstance(input_nodes, tuple):
      input_type, input_seeds = self.input_nodes
    else:
      input_type, input_seeds = None, self.input_nodes
    self._input_type = input_type

    label = self.data.get_node_label(self._input_type)
    self.input_t_label = label.to(self.device) if label is not None else None

    self._seed_loader = torch.utils.data.DataLoader(input_seeds, **kwargs)

  def __iter__(self):
    self._seeds_iter = iter(self._seed_loader)
    return self

  def __next__(self):
    raise NotImplementedError

  def _collate_fn(self, sampler_out: Union[SamplerOutput, HeteroSamplerOutput]):
    r"""format sampler output to Data/HeteroData"""
    if isinstance(sampler_out, SamplerOutput):
      x = self.data.node_features[sampler_out.node]
      y = self.input_t_label[sampler_out.node] \
        if self.input_t_label is not None else None
      if self.data.edge_features is not None and sampler_out.edge is not None:
        edge_attr = self.data.edge_features[sampler_out.edge]
      else:
        edge_attr = None
      res_data = to_data(sampler_out, batch_labels=y,
                         node_feats=x, edge_feats=edge_attr)
    else: # hetero
      x_dict = {}
      x_dict = {ntype : self.data.get_node_feature(ntype)[ids] for ntype, ids in sampler_out.node.items()}
      input_t_ids = sampler_out.node[self._input_type]
      y_dict = {self._input_type: self.input_t_label[input_t_ids]} \
        if self.input_t_label is not None else None
      edge_attr_dict = {}
      if sampler_out.edge is not None:
        for etype, eids in sampler_out.edge.items():
          efeat = self.data.get_edge_feature(etype)
          if efeat is not None:
            edge_attr_dict[etype] = efeat[eids]
      res_data = to_hetero_data(sampler_out, batch_label_dict=y_dict,
                                node_feat_dict=x_dict,
                                edge_feat_dict=edge_attr_dict,
                                edge_dir=self.data.edge_dir)
    return res_data
