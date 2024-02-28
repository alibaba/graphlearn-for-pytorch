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

from ..loader import LinkLoader

from ..data import Dataset
from ..sampler import NegativeSampling, NeighborSampler
from ..typing import NumNeighbors, InputEdges


class LinkNeighborLoader(LinkLoader):
  r"""A link-based data loader derived as an extension of the node-based
  :class:`torch_geometric.loader.NeighborLoader`.
  This loader allows for mini-batch training of GNNs on large-scale graphs
  where full-batch training is not feasible.

  More specifically, this loader first selects a sample of edges from the
  set of input edges :obj:`edge_label_index` (which may or not be edges in
  the original graph) and then constructs a subgraph from all the nodes
  present in this list by sampling :obj:`num_neighbors` neighbors in each
  iteration.

    Args:
      data (Dataset): The `graphlearn_torch.data.Dataset` object.
      num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
        number of neighbors to sample for each node in each iteration.
        In heterogeneous graphs, may also take in a dictionary denoting
        the amount of neighbors to sample for each individual edge type.
        If an entry is set to :obj:`-1`, all neighbors will be included.
      neighbor_sampler (graphlearn_torch.sampler.BaseSampler, optional):
        The sampler implementation to be used with this loader.
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
      batch_size (int): How many samples per batch to load (default: ``1``).
      shuffle (bool): Set to ``True`` to have the data reshuffled at every
        epoch (default: ``False``).
      drop_last (bool): Set to ``True`` to drop the last incomplete batch, if
        the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last
        batch will be smaller. (default: ``False``).
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
      **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
        :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
  """

  def __init__(
    self,
    data: Dataset,
    num_neighbors: NumNeighbors,
    neighbor_sampler: Optional[NeighborSampler] = None,
    edge_label_index: InputEdges = None,
    edge_label: Optional[torch.Tensor] = None,
    neg_sampling: Optional[NegativeSampling] = None,
    with_edge: bool = False,
    with_weight: bool = False,
    batch_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
    strategy: str = "random",
    device: torch.device = torch.device(0),
    seed: Optional[int] = None,
    **kwargs,
  ):
    with_neg = True if neg_sampling is not None else False
    if neighbor_sampler is None:
      neighbor_sampler = NeighborSampler(
        data.graph,
        num_neighbors=num_neighbors,
        strategy=strategy,
        with_edge=with_edge,
        with_neg=with_neg,
        with_weight=with_weight,
        device=device,
        edge_dir=data.edge_dir,
        seed=seed
      )

    super().__init__(
      data=data,
      link_sampler=neighbor_sampler,
      edge_label_index=edge_label_index,
      edge_label=edge_label,
      neg_sampling=neg_sampling,
      device=device,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=drop_last,
      edge_dir=data.edge_dir,
      **kwargs,
    )
