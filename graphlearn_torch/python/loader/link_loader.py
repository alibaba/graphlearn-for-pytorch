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


from typing import Tuple, Union, Optional, Literal

import torch

from .transform import to_data, to_hetero_data

from ..utils import convert_to_tensor
from ..data import Dataset
from ..sampler import (
  BaseSampler,
  EdgeSamplerInput,
  NegativeSampling,
  SamplerOutput,
  HeteroSamplerOutput
)
from ..typing import InputEdges, reverse_edge_type


class LinkLoader(object):
  r"""A data loader that performs mini-batch sampling from link information,
  using a generic :class:`~graphlearn_torch.sampler.BaseSampler`
  implementation that defines a
  :meth:`~graphlearn_torch.sampler.BaseSampler.sample_from_edges` function and
  is supported on the provided input :obj:`data` object.

    .. note::
      Negative sampling for triplet case is currently implemented in an
      approximate way, *i.e.* negative edges may contain false negatives.

    Args:
      data (Dataset): The `graphlearn_torch.data.Dataset` object.
      link_sampler (graphlearn_torch.sampler.BaseSampler): The sampler
        implementation to be used with this loader.
        Needs to implement
        :meth:`~graphlearn_torch.sampler.BaseSampler.sample_from_edges`.
        The sampler implementation must be compatible with the input
        :obj:`data` object.
      edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
        The edge indices, holding source and destination nodes to start
        sampling from.
        If set to :obj:`None`, all edges will be considered.
        In heterogeneous graphs, needs to be passed as a tuple that holds
        the edge type and corresponding edge indices.
        (default: :obj:`None`)
      edge_label (Tensor, optional): The labels of edge indices from which to
        start sampling from. Must be the same length as
        the :obj:`edge_label_index`. (default: :obj:`None`)
      neg_sampling (NegativeSampling, optional): The negative sampling
        configuration.
        For negative sampling mode :obj:`"binary"`, samples can be accessed
        via the attributes :obj:`edge_label_index` and :obj:`edge_label` in
        the respective edge type of the returned mini-batch.
        In case :obj:`edge_label` does not exist, it will be automatically
        created and represents a binary classification task (:obj:`0` =
        negative edge, :obj:`1` = positive edge).
        In case :obj:`edge_label` does exist, it has to be a categorical
        label from :obj:`0` to :obj:`num_classes - 1`.
        After negative sampling, label :obj:`0` represents negative edges,
        and labels :obj:`1` to :obj:`num_classes` represent the labels of
        positive edges.
        Note that returned labels are of type :obj:`torch.float` for binary
        classification (to facilitate the ease-of-use of
        :meth:`F.binary_cross_entropy`) and of type
        :obj:`torch.long` for multi-class classification (to facilitate the
        ease-of-use of :meth:`F.cross_entropy`).
        For negative sampling mode :obj:`"triplet"`, samples can be
        accessed via the attributes :obj:`src_index`, :obj:`dst_pos_index`
        and :obj:`dst_neg_index` in the respective node types of the
        returned mini-batch.
        :obj:`edge_label` needs to be :obj:`None` for :obj:`"triplet"`
        negative sampling mode.
        If set to :obj:`None`, no negative sampling strategy is applied.
        (default: :obj:`None`)
      device (torch.device, optional): The device to put the data on.
        If set to :obj:`None`, the CPU is used.
      edge_dir (str:["in", "out"]): The edge direction for sampling.
        Can be either :str:`"out"` or :str:`"in"`.
        (default: :str:`"out"`)
      **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
        :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.

  """
  def __init__(
    self,
    data: Dataset,
    link_sampler: BaseSampler,
    edge_label_index: InputEdges = None,
    edge_label: Optional[torch.Tensor] = None,
    neg_sampling: Optional[NegativeSampling] = None,
    device: torch.device = torch.device(0),
    edge_dir: Literal['out', 'in'] = 'out',
    **kwargs,
  ):
    # Get edge type (or `None` for homogeneous graphs):
    input_type, edge_label_index = get_edge_label_index(
        data, edge_label_index)
    self.data = data
    self.link_sampler = link_sampler
    self.neg_sampling = NegativeSampling.cast(neg_sampling)
    self.device = device
    self.edge_dir = edge_dir

    if (self.neg_sampling is not None and self.neg_sampling.is_binary()
        and edge_label is not None and edge_label.min() == 0):
      # Increment labels such that `zero` now denotes "negative".
      edge_label = edge_label + 1

    if (self.neg_sampling is not None and self.neg_sampling.is_triplet()
            and edge_label is not None):
      raise ValueError("'edge_label' needs to be undefined for "
                        "'triplet'-based negative sampling. Please use "
                        "`src_index`, `dst_pos_index` and "
                        "`neg_pos_index` of the returned mini-batch "
                        "instead to differentiate between positive and "
                        "negative samples.")

    self.input_data = EdgeSamplerInput(
      row=edge_label_index[0].clone(),
      col=edge_label_index[1].clone(),
      label=edge_label,
      input_type=input_type,
      neg_sampling=self.neg_sampling,
    )

    input_index = range(len(edge_label_index[0]))
    self._seed_loader = torch.utils.data.DataLoader(input_index, **kwargs)

  def __iter__(self):
    self._seeds_iter = iter(self._seed_loader)
    return self

  def __next__(self):
    seeds = self._seeds_iter._next_data().to(self.device)
    # Currently, we support the out-edge sampling manner, so we reverse the
    # direction of src and dst for the output so that features of the sampled
    # nodes during training can be aggregated from k-hop to (k-1)-hop nodes.
    sampler_out = self.link_sampler.sample_from_edges(self.input_data[seeds])
    result = self._collate_fn(sampler_out)

    return result

  def _collate_fn(self, sampler_out: Union[SamplerOutput, HeteroSamplerOutput]):
    r"""format sampler output to Data/HeteroData
      For the out-edge sampling scheme (i.e. the direction of edges in
      the output is inverse to the original graph), we put the reversed
      edge_label_index into the (dst, rev_to, src) subgraph for
      HeteroSamplerOutput and (dst, to, src) for SamplerOutput.
      However, for the in-edge sampling scheme (i.e. the direction of edges 
      in the output is the same as the original graph), we do not need to
      reverse the edge type of the sampler_out.
    """
    if isinstance(sampler_out, SamplerOutput):
      x = self.data.node_features[sampler_out.node]
      if self.data.edge_features is not None and sampler_out.edge is not None:
        edge_attr = self.data.edge_features[sampler_out.edge]
      else:
        edge_attr = None
      res_data = to_data(sampler_out,
                         node_feats=x,
                         edge_feats=edge_attr,
                        )
    else: # hetero
      x_dict = {}
      x_dict = {ntype : self.data.get_node_feature(ntype)[ids.to(torch.int64)] for ntype, ids in sampler_out.node.items()}
      edge_attr_dict = {}
      if sampler_out.edge is not None:
        for etype, eids in sampler_out.edge.items():
          if self.edge_dir == 'out':
            efeat = self.data.get_edge_feature(reverse_edge_type(etype))
          elif self.edge_dir == 'in':
            efeat = self.data.get_edge_feature(etype)
          if efeat is not None:
            edge_attr_dict[etype] = efeat[eids.to(torch.int64)]

      res_data = to_hetero_data(sampler_out,
                                node_feat_dict=x_dict,
                                edge_feat_dict=edge_attr_dict,
                                edge_dir=self.edge_dir,
                               )
    return res_data

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}()'


def get_edge_label_index(
  data: Dataset,
  edge_label_index: InputEdges
) -> Tuple[Optional[str], torch.Tensor]:
  edge_type = None
  # # Need the edge index in COO for LinkNeighborLoader:
  def _get_edge_index(edge_type):
    row, col, _, _ = data.get_graph(edge_type).topo.to_coo()
    return (row, col)

  if not isinstance(edge_label_index, Tuple):
    if edge_label_index is None:
      return None, _get_edge_index(edge_type)
    return None, convert_to_tensor(edge_label_index)

  if isinstance(edge_label_index[0], str):
    edge_type = edge_label_index

    return edge_type, _get_edge_index(edge_type)

  assert len(edge_label_index) == 2
  edge_type, edge_label_index = convert_to_tensor(edge_label_index)

  if edge_label_index is None:
    row, col, _, _ = data.get_graph(edge_type).topo.to_coo()
    return edge_type, (row, col)

  return edge_type, edge_label_index