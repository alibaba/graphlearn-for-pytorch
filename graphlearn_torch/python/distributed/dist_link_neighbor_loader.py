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

from typing import Optional, Literal

import torch

from ..sampler import (
  EdgeSamplerInput, SamplingType, SamplingConfig, NegativeSampling
)
from ..loader import get_edge_label_index
from ..typing import InputEdges, NumNeighbors

from .dist_dataset import DistDataset
from .dist_options import AllDistSamplingWorkerOptions
from .dist_loader import DistLoader


class DistLinkNeighborLoader(DistLoader):
  r""" A distributed loader that preform sampling from edges.

  Args:
    data (DistDataset, optional): The ``DistDataset`` object of a partition of
      graph data and feature data, along with distributed patition books. The
      input dataset must be provided in non-server distribution mode.
    num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]):
      The number of neighbors to sample for each node in each iteration.
      In heterogeneous graphs, may also take in a dictionary denoting
      the amount of neighbors to sample for each individual edge type.
    batch_size (int): How many samples per batch to load (default: ``1``).
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
    shuffle (bool): Set to ``True`` to have the data reshuffled at every
      epoch (default: ``False``).
    drop_last (bool): Set to ``True`` to drop the last incomplete batch, if
      the dataset size is not divisible by the batch size. If ``False`` and
      the size of dataset is not divisible by the batch size, then the last
      batch will be smaller. (default: ``False``).
    with_edge (bool): Set to ``True`` to sample with edge ids and also include
      them in the sampled results. (default: ``False``).
    edge_dir (str:["in", "out"]): The edge direction for sampling.
      Can be either :str:`"out"` or :str:`"in"`.
      (default: :str:`"out"`)
    collect_features (bool): Set to ``True`` to collect features for nodes
      of each sampled subgraph. (default: ``False``).
    to_device (torch.device, optional): The target device that the sampled
      results should be copied to. If set to ``None``, the current cuda device
      (got by ``torch.cuda.current_device``) will be used if available,
      otherwise, the cpu device will be used. (default: ``None``).
    worker_options (optional): The options for launching sampling workers.
      (1) If set to ``None`` or provided with a ``CollocatedDistWorkerOptions``
      object, a single collocated sampler will be launched on the current
      process, while the separate sampling mode will be disabled . (2) If
      provided with a ``MpDistWorkerOptions`` object, the sampling workers will
      be launched on spawned subprocesses, and a share-memory based channel
      will be created for sample message passing from multiprocessing workers
      to the current loader. (3) If provided with a ``RemoteDistWorkerOptions``
      object, the sampling workers will be launched on remote sampling server
      nodes, and a remote channel will be created for cross-machine message
      passing. (default: ``None``).
  """
  def __init__(self,
               data: Optional[DistDataset],
               num_neighbors: NumNeighbors,
               batch_size: int = 1,
               edge_label_index: InputEdges = None,
               edge_label: Optional[torch.Tensor] = None,
               neg_sampling: Optional[NegativeSampling] = None,
               shuffle: bool = False,
               drop_last: bool = False,
               with_edge: bool = False,
               with_weight: bool = False,
               edge_dir: Literal['in', 'out'] = 'out',
               collect_features: bool = False,
               to_device: Optional[torch.device] = None,
               random_seed: Optional[int] = None,
               worker_options: Optional[AllDistSamplingWorkerOptions] = None):
    # Get edge type (or `None` for homogeneous graphs):
    input_type, edge_label_index = get_edge_label_index(
        data, edge_label_index)
    with_neg = neg_sampling is not None
    self.neg_sampling = NegativeSampling.cast(neg_sampling)

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

    input_data = EdgeSamplerInput(
      row=edge_label_index[0].clone(),
      col=edge_label_index[1].clone(),
      label=edge_label,
      input_type=input_type,
      neg_sampling=self.neg_sampling,
    )

    sampling_config = SamplingConfig(
      SamplingType.LINK, num_neighbors, batch_size, shuffle,
      drop_last, with_edge, collect_features, with_neg, 
      with_weight=with_weight, edge_dir=edge_dir, seed=random_seed
    )

    super().__init__(
      data, input_data, sampling_config, to_device, worker_options
    )
