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

from typing import List, Optional, Union
import concurrent
import torch
from torch_geometric.data import Data, HeteroData

from ..channel import SampleMessage, ShmChannel, RemoteReceivingChannel
from ..loader import to_data, to_hetero_data
from ..sampler import (
  NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput, SamplerOutput,
  HeteroSamplerOutput, SamplingConfig, SamplingType
)
from ..typing import (NodeType, EdgeType, as_str, reverse_edge_type)
from ..utils import get_available_device, ensure_device, python_exit_status

from .dist_client import request_server
from .dist_context import get_context
from .dist_dataset import DistDataset
from .dist_options import (
  CollocatedDistSamplingWorkerOptions,
  MpDistSamplingWorkerOptions,
  RemoteDistSamplingWorkerOptions,
  AllDistSamplingWorkerOptions,
)
from .dist_sampling_producer import (
  DistMpSamplingProducer, DistCollocatedSamplingProducer
)
from .dist_server import DistServer
from .rpc import rpc_is_initialized


class DistLoader(object):
  r""" A generic data loader base that performs distributed sampling, which
  allows mini-batch training of GNNs on large-scale graphs when full-batch
  training is not feasible.

  This loader supports launching a collocated sampling worker on the current
  process, or launching separate sampling workers on the spawned subprocesses
  or remote server nodes. When using the separate sampling mode, a worker group
  including the information of separate sampling workers should be provided.

  Note that the separate sampling mode supports asynchronous and concurrent
  sampling on each separate worker, which will achieve better performance
  and is recommended to use. If you want to use a collocated sampling worker,
  all sampling for each seed batch will be blocking and synchronous.

  When launching a collocated sampling worker or some multiprocessing sampling
  workers (on spwaned subprocesses), the distribution mode must be non-server
  and only contains a group of parallel worker processes, this means that the
  graph and feature store should be partitioned among all those parallel worker
  processes and managed by them, sampling and training tasks will run on each
  worker process at the same time.

  Otherwise, when launching some remote sampling workers, the distribution mode
  must be a server-client framework, which contains a group of server workers
  and a group of client workers, the graph and feature store will be partitioned
  and managed by all server workers. All client workers are responsible for
  training tasks and launch some workers on remote servers to perform sampling
  tasks, the sampled results will be consumed by client workers with a remote
  message channel.

  Args:
    data (DistDataset, optional): The ``DistDataset`` object of a partition of
      graph data and feature data, along with distributed patition books. The
      input dataset must be provided in non-server distribution mode.
    input_data (NodeSamplerInput or EdgeSamplerInput or RemoteSamplerInput): 
      The input data for which neighbors or subgraphs are sampled to create 
      mini-batches. In heterogeneous graphs, needs to be passed as a tuple that
      holds the node type and node indices.
    sampling_config (SamplingConfig): The Configuration info for sampling.
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

  def __init__(
    self,
    data: Optional[DistDataset],
    input_data: Union[NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput,
                      List[RemoteSamplerInput]],
    sampling_config: SamplingConfig,
    to_device: Optional[torch.device] = None,
    worker_options: Optional[AllDistSamplingWorkerOptions] = None
  ):
    self.data = data
    self.input_data = input_data
    self.sampling_type = sampling_config.sampling_type
    self.num_neighbors = sampling_config.num_neighbors
    self.batch_size = sampling_config.batch_size
    self.shuffle = sampling_config.shuffle
    self.drop_last = sampling_config.drop_last
    self.with_edge = sampling_config.with_edge
    self.with_weight = sampling_config.with_weight
    self.collect_features = sampling_config.collect_features
    self.edge_dir = sampling_config.edge_dir
    self.sampling_config = sampling_config
    self.to_device = get_available_device(to_device)
    self.worker_options = worker_options
    if self.worker_options is None:
      self.worker_options = CollocatedDistSamplingWorkerOptions()

    self._is_collocated_worker = isinstance(
      self.worker_options, CollocatedDistSamplingWorkerOptions
    )
    self._is_mp_worker = isinstance(
      self.worker_options, MpDistSamplingWorkerOptions
    )
    self._is_remote_worker = isinstance(
      self.worker_options, RemoteDistSamplingWorkerOptions
    )

    if self.data is not None:
      self.num_data_partitions = self.data.num_partitions
      self.data_partition_idx = self.data.partition_idx
      self._set_ntypes_and_etypes(
        self.data.get_node_types(), self.data.get_edge_types()
      )

    self._num_recv = 0
    self._epoch = 0

    current_ctx = get_context()
    if current_ctx is None:
      raise RuntimeError(
        f"'{self.__class__.__name__}': the distributed "
        f"context of has not been initialized."
      )

    if self._is_remote_worker:
      if not current_ctx.is_client():
        raise RuntimeError(
          f"'{self.__class__.__name__}': `DistNeighborLoader` "
          f"must be used on a client worker process."
        )
      self._num_expected = float(
        'inf'
      )  # for remote worker, end of epoch is determined by server

      # Launch remote sampling workers
      self._with_channel = True

      self._server_rank_list = self.worker_options.server_rank \
        if isinstance(self.worker_options.server_rank, List) else [self.worker_options.server_rank]
      self._input_data_list = self.input_data \
        if isinstance(self.input_data, List) else [self.input_data]

      self._input_type = self._input_data_list[0].input_type

      self.num_data_partitions, self.data_partition_idx, ntypes, etypes = \
        request_server(self._server_rank_list[0], DistServer.get_dataset_meta)
      self._set_ntypes_and_etypes(ntypes, etypes)

      self._producer_id_list = []
      futures = []
      for input_data in self._input_data_list:
        if not isinstance(input_data, RemoteSamplerInput):
          input_data = input_data.to(torch.device('cpu'))

      with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(request_server, server_rank, DistServer.create_sampling_producer, input_data, self.sampling_config, self.worker_options) \
                         for server_rank, input_data in zip(self._server_rank_list, self._input_data_list)]
        
      for future in futures:
        producer_id = future.result()
        self._producer_id_list.append(producer_id)
      self._channel = RemoteReceivingChannel(
        self._server_rank_list, self._producer_id_list,
        self.worker_options.prefetch_size
      )
    else:
      self._input_len = len(self.input_data)
      self._input_type = self.input_data.input_type
      self._num_expected = self._input_len // self.batch_size
      if not self.drop_last and self._input_len % self.batch_size != 0:
        self._num_expected += 1

      if self._is_collocated_worker:
        if not current_ctx.is_worker():
          raise RuntimeError(
            f"'{self.__class__.__name__}': only supports "
            f"launching a collocated sampler with a non-server "
            f"distribution mode, current role of distributed "
            f"context is {current_ctx.role}."
          )
        if self.data is None:
          raise ValueError(
            f"'{self.__class__.__name__}': missing input dataset "
            f"when launching a collocated sampler."
          )

        # Launch collocated producer
        self._with_channel = False

        self._collocated_producer = DistCollocatedSamplingProducer(
          self.data, self.input_data, self.sampling_config, self.worker_options,
          self.to_device
        )
        self._collocated_producer.init()

      elif self._is_mp_worker:
        if not current_ctx.is_worker():
          raise RuntimeError(
            f"'{self.__class__.__name__}': only supports "
            f"launching multiprocessing sampling workers with "
            f"a non-server distribution mode, current role of "
            f"distributed context is {current_ctx.role}."
          )
        if self.data is None:
          raise ValueError(
            f"'{self.__class__.__name__}': missing input dataset "
            f"when launching multiprocessing sampling workers."
          )

        # Launch multiprocessing sampling workers
        self._with_channel = True
        self.worker_options._set_worker_ranks(current_ctx)

        self._channel = ShmChannel(
          self.worker_options.channel_capacity, self.worker_options.channel_size
        )
        if self.worker_options.pin_memory:
          self._channel.pin_memory()

        self._mp_producer = DistMpSamplingProducer(
          self.data, self.input_data, self.sampling_config, self.worker_options,
          self._channel
        )
        self._mp_producer.init()
      else:
        raise ValueError(
          f"'{self.__class__.__name__}': found invalid "
          f"worker options type '{type(worker_options)}'"
        )

    self._shutdowned = False

  def __del__(self):
    if python_exit_status is True or python_exit_status is None:
      return
    self.shutdown()

  def shutdown(self):
    if self._shutdowned:
      return
    if self._is_collocated_worker:
      self._collocated_producer.shutdown()
    elif self._is_mp_worker:
      self._mp_producer.shutdown()
    else:
      if rpc_is_initialized() is True:
        for server_rank, producer_id in zip(self._server_rank_list, self._producer_id_list):
          request_server(
            server_rank, DistServer.destroy_sampling_producer,
            producer_id
          )
    self._shutdowned = True

  def __next__(self):
    if self._num_recv == self._num_expected:
      raise StopIteration

    if self._with_channel:
      msg = self._channel.recv()
    else:
      msg = self._collocated_producer.sample()

    result = self._collate_fn(msg)
    self._num_recv += 1
    return result

  def __iter__(self):
    self._num_recv = 0
    if self._is_collocated_worker:
      self._collocated_producer.reset()
    elif self._is_mp_worker:
      self._mp_producer.produce_all()
    else:
      for server_rank, producer_id in zip(self._server_rank_list, self._producer_id_list):
        request_server(
          server_rank, 
          DistServer.start_new_epoch_sampling,
          producer_id,
          self._epoch
        )
      self._channel.reset()
    self._epoch += 1
    return self

  def _set_ntypes_and_etypes(self, node_types: List[NodeType],
                             edge_types: List[EdgeType]):
    self._node_types = node_types
    self._edge_types = edge_types
    self._reversed_edge_types = []
    self._etype_str_to_rev = {}
    if self._edge_types is not None:
      for etype in self._edge_types:
        rev_etype = reverse_edge_type(etype)
        if self.edge_dir == 'out':
          self._reversed_edge_types.append(rev_etype)
          self._etype_str_to_rev[as_str(etype)] = rev_etype
        elif self.edge_dir == 'in':
          self._reversed_edge_types.append(etype)
          self._etype_str_to_rev[as_str(rev_etype)] = etype

  def _collate_fn(
    self,
    msg: SampleMessage
  ) -> Union[Data, HeteroData]:
    r""" Collate sampled messages as PyG's Data/HeteroData
    """
    ensure_device(self.to_device)
    is_hetero = bool(msg['#IS_HETERO'])

    # extract meta data
    metadata = {}
    for k in msg.keys():
      if k.startswith('#META.'):
        meta_key = str(k[6:])
        metadata[meta_key] = msg[k].to(self.to_device)
    if len(metadata) == 0:
      metadata = None

    # Heterogeneous sampling results
    if is_hetero:
      node_dict, row_dict, col_dict, edge_dict = {}, {}, {}, {}
      nfeat_dict, efeat_dict = {}, {}
      num_sampled_nodes_dict, num_sampled_edges_dict = {}, {}

      for ntype in self._node_types:
        ids_key = f'{as_str(ntype)}.ids'
        if ids_key in msg:
          node_dict[ntype] = msg[ids_key].to(self.to_device)
        nfeat_key = f'{as_str(ntype)}.nfeats'
        if nfeat_key in msg:
          nfeat_dict[ntype] = msg[nfeat_key].to(self.to_device)
        num_sampled_nodes_key = f'{as_str(ntype)}.num_sampled_nodes'
        if num_sampled_nodes_key in msg:
          num_sampled_nodes_dict[ntype] = msg[num_sampled_nodes_key]

      for etype_str, rev_etype in self._etype_str_to_rev.items():
        rows_key = f'{etype_str}.rows'
        cols_key = f'{etype_str}.cols'
        if rows_key in msg:
          # The edge index should be reversed.
          row_dict[rev_etype] = msg[cols_key].to(self.to_device)
          col_dict[rev_etype] = msg[rows_key].to(self.to_device)
        eids_key = f'{etype_str}.eids'
        if eids_key in msg:
          edge_dict[rev_etype] = msg[eids_key].to(self.to_device)
        num_sampled_edges_key = f'{etype_str}.num_sampled_edges'
        if num_sampled_edges_key in msg:
          num_sampled_edges_dict[rev_etype] = msg[num_sampled_edges_key]
        efeat_key = f'{etype_str}.efeats'
        if efeat_key in msg:
          efeat_dict[rev_etype] = msg[efeat_key].to(self.to_device)

      if len(nfeat_dict) == 0:
        nfeat_dict = None
      if len(efeat_dict) == 0:
        efeat_dict = None

      if self.sampling_config.sampling_type in [SamplingType.NODE,
                                                SamplingType.SUBGRAPH]:
        batch_dict = {
          self._input_type: node_dict[self._input_type][:self.batch_size]
        }
        batch_labels_key = f'{self._input_type}.nlabels'
        if batch_labels_key in msg:
          batch_labels = msg[batch_labels_key].to(self.to_device)
        else:
          batch_labels = None
        batch_label_dict = {self._input_type: batch_labels}
      else:
        batch_dict = {}
        batch_label_dict = {}

      output = HeteroSamplerOutput(node_dict, row_dict, col_dict,
                                   edge_dict if len(edge_dict) else None,
                                   batch_dict,
                                   num_sampled_nodes=num_sampled_nodes_dict,
                                   num_sampled_edges=num_sampled_edges_dict,
                                   edge_types=self._reversed_edge_types,
                                   input_type=self._input_type,
                                   device=self.to_device,
                                   metadata=metadata)
      res_data = to_hetero_data(
        output, batch_label_dict, nfeat_dict, efeat_dict, self.edge_dir)

    # Homogeneous sampling results
    else:
      ids = msg['ids'].to(self.to_device)
      rows = msg['rows'].to(self.to_device)
      cols = msg['cols'].to(self.to_device)
      eids = msg['eids'].to(self.to_device) if 'eids' in msg else None
      num_sampled_nodes = msg['num_sampled_nodes'] if 'num_sampled_nodes' in msg else None
      num_sampled_edges = msg['num_sampled_edges'] if 'num_sampled_edges' in msg else None

      nfeats = msg['nfeats'].to(self.to_device) if 'nfeats' in msg else None
      efeats = msg['efeats'].to(self.to_device) if 'efeats' in msg else None

      if self.sampling_config.sampling_type in [SamplingType.NODE,
                                                SamplingType.SUBGRAPH]:
        batch = ids[:self.batch_size]
        batch_labels = msg['nlabels'].to(self.to_device) if 'nlabels' in msg else None
      else:
        batch = None
        batch_labels = None

      # The edge index should be reversed.
      output = SamplerOutput(ids, cols, rows, eids, batch,
                             num_sampled_nodes, num_sampled_edges,
                             device=self.to_device, metadata=metadata)
      res_data = to_data(output, batch_labels, nfeats, efeats)

    return res_data
