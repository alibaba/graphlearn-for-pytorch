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

import logging
import time
import threading
from typing import Dict, Optional, Union
import warnings

from ..channel import ShmChannel, QueueTimeoutError
from ..sampler import NodeSamplerInput, EdgeSamplerInput, SamplingConfig, RemoteSamplerInput

from .dist_context import get_context, _set_server_context
from .dist_dataset import DistDataset
from .dist_options import RemoteDistSamplingWorkerOptions
from .dist_sampling_producer import DistMpSamplingProducer
from .rpc import barrier, init_rpc, shutdown_rpc


SERVER_EXIT_STATUS_CHECK_INTERVAL = 5.0
r""" Interval (in seconds) to check exit status of server.
"""

class DistServer(object):
  r""" A server that supports launching remote sampling workers for
  training clients.

  Note that this server is enabled only when the distribution mode is a
  server-client framework, and the graph and feature store will be partitioned
  and managed by all server nodes.

  Args:
    dataset (DistDataset): The ``DistDataset`` object of a partition of graph
      data and feature data, along with distributed patition books.
  """
  def __init__(self, dataset: DistDataset):
    self.dataset = dataset
    self._lock = threading.RLock()
    self._exit = False
    self._cur_producer_idx = 0 # auto incremental index (same as producer count)
    # The mapping from the key in worker options (such as 'train', 'test')
    # to producer id
    self._worker_key2producer_id: Dict[str, int] = {}  
    self._producer_pool: Dict[int, DistMpSamplingProducer] = {}
    self._msg_buffer_pool: Dict[int, ShmChannel] = {}
    self._epoch: Dict[int, int] = {} # last epoch for the producer

  def shutdown(self):
    for producer_id in list(self._producer_pool.keys()):
      self.destroy_sampling_producer(producer_id)
    assert len(self._producer_pool) == 0
    assert len(self._msg_buffer_pool) == 0

  def wait_for_exit(self):
    r""" Block until the exit flag been set to ``True``.
    """
    while not self._exit:
      time.sleep(SERVER_EXIT_STATUS_CHECK_INTERVAL)

  def exit(self):
    r""" Set the exit flag to ``True``.
    """
    self._exit = True
    return self._exit

  def get_dataset_meta(self):
    r""" Get the meta info of the distributed dataset managed by the current
    server, including partition info and graph types.
    """
    return self.dataset.num_partitions, self.dataset.partition_idx, \
      self.dataset.get_node_types(), self.dataset.get_edge_types()

  def create_sampling_producer(
    self,
    sampler_input: Union[NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput],
    sampling_config: SamplingConfig,
    worker_options: RemoteDistSamplingWorkerOptions,
  ) -> int:
    r""" Create and initialize an instance of ``DistSamplingProducer`` with
    a group of subprocesses for distributed sampling.

    Args:
      sampler_input (NodeSamplerInput or EdgeSamplerInput): The input data
        for sampling.
      sampling_config (SamplingConfig): Configuration of sampling meta info.
      worker_options (RemoteDistSamplingWorkerOptions): Options for launching
        remote sampling workers by this server.

    Returns:
      A unique id of created sampling producer on this server.
    """
    if isinstance(sampler_input, RemoteSamplerInput):
      sampler_input = sampler_input.to_local_sampler_input(dataset=self.dataset)
    
    with self._lock: 
      producer_id = self._worker_key2producer_id.get(worker_options.worker_key)
      if producer_id is None:
        producer_id = self._cur_producer_idx
        self._worker_key2producer_id[worker_options.worker_key] = producer_id
        self._cur_producer_idx += 1
        buffer = ShmChannel(
          worker_options.buffer_capacity, worker_options.buffer_size
        )
        producer = DistMpSamplingProducer(
          self.dataset, sampler_input, sampling_config, worker_options, buffer
        )
        producer.init()
        self._producer_pool[producer_id] = producer
        self._msg_buffer_pool[producer_id] = buffer
        self._epoch[producer_id] = -1
    return producer_id

  def destroy_sampling_producer(self, producer_id: int):
    r""" Shutdown and destroy a sampling producer managed by this server with
    its producer id.
    """
    with self._lock:
      producer = self._producer_pool.get(producer_id, None)
      if producer is not None:
        producer.shutdown()
        self._producer_pool.pop(producer_id)
        self._msg_buffer_pool.pop(producer_id)
        self._epoch.pop(producer_id)

  def start_new_epoch_sampling(self, producer_id: int, epoch: int):
    r""" Start a new epoch sampling tasks for a specific sampling producer
    with its producer id.
    """
    with self._lock:
      cur_epoch = self._epoch[producer_id]
      if cur_epoch < epoch:
        self._epoch[producer_id] = epoch
        producer = self._producer_pool.get(producer_id, None)
        if producer is not None:
          producer.produce_all()

  def fetch_one_sampled_message(self, producer_id: int):
    r""" Fetch a sampled message from the buffer of a specific sampling
    producer with its producer id.
    """
    producer = self._producer_pool.get(producer_id, None)
    if producer is None:
      warnings.warn('invalid producer_id {producer_id}')
      return None, False
    if producer.is_all_sampling_completed_and_consumed():
      return None, True
    buffer = self._msg_buffer_pool.get(producer_id, None)
    while True:
        try:
            msg = buffer.recv(timeout_ms=500)
            return msg, False
        except QueueTimeoutError as e:
            if producer.is_all_sampling_completed():
                return None, True


_dist_server: DistServer = None
r""" ``DistServer`` instance of the current process.
"""


def get_server() -> DistServer:
  r""" Get the ``DistServer`` instance on the current process.
  """
  return _dist_server


def init_server(num_servers: int, server_rank: int, dataset: DistDataset,
                master_addr: str, master_port: int, num_clients: int = 0,
                num_rpc_threads: int = 16, request_timeout: int = 180,
                server_group_name: Optional[str] = None, is_dynamic: bool = False):
  r""" Initialize the current process as a server and establish connections
  with all other servers and clients. Note that this method should be called
  only in the server-client distribution mode.

  Args:
    num_servers (int): Number of processes participating in the server group.
    server_rank (int): Rank of the current process withing the server group (it
      should be a number between 0 and ``num_servers``-1).
    dataset (DistDataset): The ``DistDataset`` object of a partition of graph
      data and feature data, along with distributed patition book info.
    master_addr (str): The master TCP address for RPC connection between all
      servers and clients, the value of this parameter should be same for all
      servers and clients.
    master_port (int): The master TCP port for RPC connection between all
      servers and clients, the value of this parameter should be same for all
      servers and clients.
    num_clients (int): Number of processes participating in the client group.
      if ``is_dynamic`` is ``True``, this parameter will be ignored.
    num_rpc_threads (int): The number of RPC worker threads used for the
      current server to respond remote requests. (Default: ``16``).
    request_timeout (int): The max timeout seconds for remote requests,
      otherwise an exception will be raised. (Default: ``16``).
    server_group_name (str): A unique name of the server group that current
      process belongs to. If set to ``None``, a default name will be used.
      (Default: ``None``).
    is_dynamic (bool): Whether the world size is dynamic. (Default: ``False``).
  """
  if server_group_name:
    server_group_name = server_group_name.replace('-', '_')
  _set_server_context(num_servers, server_rank, server_group_name, num_clients)
  global _dist_server
  _dist_server = DistServer(dataset=dataset)
  init_rpc(master_addr, master_port, num_rpc_threads, request_timeout, is_dynamic=is_dynamic)


def wait_and_shutdown_server():
  r""" Block until all client have been shutdowned, and further shutdown the
  server on the current process and destroy all RPC connections.
  """
  current_context = get_context()
  if current_context is None:
    logging.warning("'wait_and_shutdown_server': try to shutdown server when "
                    "the current process has not been initialized as a server.")
    return
  if not current_context.is_server():
    raise RuntimeError(f"'wait_and_shutdown_server': role type of "
                       f"the current process context is not a server, "
                       f"got {current_context.role}.")
  global _dist_server
  _dist_server.wait_for_exit()
  _dist_server.shutdown()
  _dist_server = None
  barrier()
  shutdown_rpc()


def _call_func_on_server(func, *args, **kwargs):
  r""" A callee entry for remote requests on the server side.
  """
  if not callable(func):
    logging.warning(f"'_call_func_on_server': receive a non-callable "
                    f"function target {func}")
    return None

  server = get_server()
  if hasattr(server, func.__name__):
    return func(server, *args, **kwargs)

  return func(*args, **kwargs)
