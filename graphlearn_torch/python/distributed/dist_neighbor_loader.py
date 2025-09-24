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

from typing import Optional, Literal, List

import torch

from ..sampler import (
    AllNodeSamplerInputs,
    NodeSamplerInput,
    SamplingType,
    SamplingConfig,
    RemoteNodeSplitSamplerInput,
    RemoteNodePathSamplerInput,
)
from ..typing import InputNodes, NumNeighbors, Split

from .dist_dataset import DistDataset
from .dist_options import AllDistSamplingWorkerOptions, RemoteDistSamplingWorkerOptions
from .dist_loader import DistLoader


class DistNeighborLoader(DistLoader):
  r"""A distributed loader that preform sampling from nodes.

  Args:
    data (DistDataset, optional): The ``DistDataset`` object of a partition of
      graph data and feature data, along with distributed patition books. The
      input dataset must be provided in non-server distribution mode.
    num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]):
      The number of neighbors to sample for each node in each iteration.
      In heterogeneous graphs, may also take in a dictionary denoting
      the amount of neighbors to sample for each individual edge type.
    input_nodes (torch.Tensor or Tuple[str, torch.Tensor], optional): The node seeds for
      which neighbors are sampled to create mini-batches. In heterogeneous
      graphs, needs to be passed as a tuple that holds the node type and
      node seeds.
      Must *not* be provided when ``node_sampler_input`` is provided.
    node_sampler_input: (AllNodeSamplerInputs, optional): The input for node sampler.
      May be used to override the functionality when sampling.
      Must *not* be provided when ``input_nodes`` is provided.
    batch_size (int): How many samples per batch to load (default: ``1``).
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
               input_nodes: Optional[InputNodes] = None,
               node_sampler_input: Optional[AllNodeSamplerInputs] = None,
               batch_size: int = 1,
               shuffle: bool = False,
               drop_last: bool = False,
               with_edge: bool = False,
               with_weight: bool = False,
               edge_dir: Literal['in', 'out'] = 'out',
               collect_features: bool = False,
               to_device: Optional[torch.device] = None,
               random_seed: Optional[int] = None,
               worker_options: Optional[AllDistSamplingWorkerOptions] = None):
    if (input_nodes is None) == (node_sampler_input is None):
      raise ValueError(
        f"Exactly one of input_nodes or node_sampler_input must be provided. Received input_nodes: {type(input_nodes)}, node_sampler_input: {type(node_sampler_input)}"
      )
    if isinstance(input_nodes, tuple):
      input_type, input_seeds = input_nodes
    elif input_nodes is not None:
      input_type, input_seeds = None, input_nodes

    if node_sampler_input is not None:
      _check_input_type(node_sampler_input, worker_options)
      input_data = node_sampler_input
    elif isinstance(worker_options, RemoteDistSamplingWorkerOptions):
      if isinstance(input_seeds, Split):
        input_data = RemoteNodeSplitSamplerInput(split=input_seeds, input_type=input_type)
        if isinstance(worker_options.server_rank, List):
          input_data = [input_data] * len(worker_options.server_rank)
      elif isinstance(input_seeds, List):
        input_data = []
        for elem in input_seeds:
          input_data.append(RemoteNodePathSamplerInput(node_path=elem, input_type=input_type))
      elif isinstance(input_seeds, str):
        input_data = RemoteNodePathSamplerInput(node_path=input_seeds, input_type=input_type)
      else:
        raise ValueError("Invalid input seeds")
    else:
      input_data = NodeSamplerInput(node=input_seeds, input_type=input_type)

    sampling_config = SamplingConfig(
      SamplingType.NODE, num_neighbors, batch_size, shuffle,
      drop_last, with_edge, collect_features, with_neg=False, 
      with_weight=with_weight, edge_dir=edge_dir, seed=random_seed
    )

    super().__init__(data, input_data, sampling_config, to_device, worker_options)


def _check_input_type(node_sampler_input: AllNodeSamplerInputs, worker_options: AllDistSamplingWorkerOptions):
    if isinstance(worker_options, RemoteDistSamplingWorkerOptions):
        if isinstance(worker_options.server_rank, List):
            if not isinstance(node_sampler_input, List):
                raise ValueError(
                    f"When worker options is {type(RemoteDistSamplingWorkerOptions)} with server_rank as List, node_sampler_input must be a list, but got {type(node_sampler_input)}"
                )
            if not all(
                isinstance(sampler_input, RemoteNodeSplitSamplerInput)
                for sampler_input in node_sampler_input
            ):
                raise ValueError(
                    f"When worker options is RemoteDistSamplingWorkerOptions with server_rank as List, node_sampler_input must be a list and all elements of the list must be of type RemoteNodePathSamplerInput, but got {[type(nsi) for nsi in node_sampler_input]}"
                )
        elif isinstance(node_sampler_input, List):
          if not all(isinstance(sampler_input, RemoteNodePathSamplerInput) for sampler_input in node_sampler_input):
            raise ValueError(
                f"When worker options is {type(worker_options)} and node_sampler_input is a list, all elements of the list must be of type RemoteNodePathSamplerInput, but got {[type(nsi) for nsi in node_sampler_input]}"
            )
        elif not isinstance(node_sampler_input, RemoteNodePathSamplerInput):  
            raise ValueError(
                f"When worker options is {type(worker_options)} and node_sampler_input is not a list, node_sampler_input must be of type RemoteNodePathSamplerInput, but got {type(node_sampler_input)}"
            )