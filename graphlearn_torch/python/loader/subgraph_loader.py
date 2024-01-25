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

from typing import Optional

import torch

from . import NodeLoader

from ..data import Dataset
from ..sampler import NeighborSampler, NodeSamplerInput
from ..typing import InputNodes, NumNeighbors


class SubGraphLoader(NodeLoader):
  r""" A distributed loader for subgraph sampling.

  Args:
    data (Dataset): The `graphlearn_torch.data.Dataset` object.
    num_neighbors (List[int]): The
      number of neighbors to sample for each node in each iteration.
      If an entry is set to :obj:`-1`, all neighbors will be included.
    input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
      indices of nodes for which neighbors are sampled to create
      mini-batches.
      Needs to be either given as a :obj:`torch.LongTensor` or
      :obj:`torch.BoolTensor`.
      In heterogeneous graphs, needs to be passed as a tuple that holds
      the node type and node indices.
    batch_size (int): How many samples per batch to load (default: ``1``).
    shuffle (bool): Set to ``True`` to have the data reshuffled at every
      epoch (default: ``False``).
    drop_last (bool): Set to ``True`` to drop the last incomplete batch, if
      the dataset size is not divisible by the batch size. If ``False`` and
      the size of dataset is not divisible by the batch size, then the last
      batch will be smaller. (default: ``False``).
    with_edge (bool): Set to ``True`` to sample with edge ids and also include
      them in the sampled results. (default: ``False``).
    strategy: (str): Set sampling strategy for the default neighbor sampler
      provided by graphlearn-torch. (default: ``"random"``).
  """
  def __init__(
    self,
    data: Dataset,
    input_nodes: InputNodes,
    num_neighbors: Optional[NumNeighbors] = None,
    neighbor_sampler: Optional[NeighborSampler] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
    with_edge: bool = False,
    strategy: str = 'random',
    device: torch.device = torch.device(0),
    seed: Optional[int] = None,
    **kwargs
  ):
    if neighbor_sampler is None:
      neighbor_sampler = NeighborSampler(
        data.graph,
        num_neighbors=num_neighbors,
        strategy=strategy,
        with_edge=with_edge,
        device=device,
        seed=seed
      )

    super().__init__(
      data=data,
      node_sampler=neighbor_sampler,
      input_nodes=input_nodes,
      device=device,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=drop_last,
      **kwargs,
    )

  def __next__(self):
    seeds = self._seeds_iter._next_data().to(self.device)
    inputs = NodeSamplerInput(
      node=seeds,
      input_type=self._input_type
    )
    out = self.sampler.subgraph(inputs)
    result = self._collate_fn(out)
    result.mapping = result.metadata

    return result
