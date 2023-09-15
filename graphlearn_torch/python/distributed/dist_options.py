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

import os
from typing import List, Optional, Union, Literal

import torch

from ..utils import assign_device

from .dist_context import DistContext, assign_server_by_order


class _BasicDistSamplingWorkerOptions(object):
  r""" Basic options to launch distributed sampling workers.

  Args:
    num_workers (int): How many workers to use for distributed neighbor
      sampling of the current process, must be same for each process of
      the current context group. (default: ``1``).
    worker_devices (torch.device or List[torch.device], optional): List of
      devices assgined to workers of this group. If set to ``None``, the
      devices to use will be automatically assigned (the cuda device will be
      preferred if available). (default: ``None``).
    worker_concurrency (int): The max sampling concurrency with different
      seeds batches for each sampling worker, which should not exceed 32.
      (default: ``1``).
    master_addr (str, optional): Master address for rpc initialization across
      all sampling workers. the environment varaible ``MASTER_ADDR`` will be
      used if set to ``None``. (default: ``None``).
    master_port (str or int, optional): Master port for rpc initialization
      across all sampling workers. If set to ``None``, in order to avoid
      conflicts with master port already used by other modules (e.g., the
      method ``init_process_group`` of ``torch.distributed``), the value of
      environment varaible ``MASTER_PORT`` will be increased by one as the
      real rpc port for sampling workers. Otherwise, the provided port should
      be guaranteed to avoid such conflicts. (default: ``None``).
    num_rpc_threads (int, optional): Number of threads used for rpc agent on
      each sampling worker. If set to ``None``, the number of rpc threads to
      use will be specified according to the actual workload, but will not
      exceed 16. (default: ``None``).
    rpc_timeout (float): The timeout in seconds for all rpc requests during
      distributed sampling and feature collection. (default: ``180``).
  """
  def __init__(self,
               num_workers: int = 1,
               worker_devices: Optional[List[torch.device]] = None,
               worker_concurrency: int = 1,
               master_addr: Optional[str] = None,
               master_port: Optional[Union[str, int]] = None,
               num_rpc_threads: Optional[int] = None,
               rpc_timeout: float = 180):
    self.num_workers = num_workers

    # Not sure yet, will be calculated later.
    self.worker_world_size = None
    self.worker_ranks = None

    if worker_devices is None:
      self.worker_devices = None
    elif isinstance(worker_devices, list) or isinstance(worker_devices, tuple):
      assert len(worker_devices) == self.num_workers
      self.worker_devices = list(worker_devices)
    else:
      self.worker_devices = [worker_devices] * self.num_workers

    # Worker concurrency should not exceed 32.
    self.worker_concurrency = max(worker_concurrency, 1)
    self.worker_concurrency = min(self.worker_concurrency, 32)

    if master_addr is not None:
      self.master_addr = str(master_addr)
    elif os.environ.get('MASTER_ADDR') is not None:
      self.master_addr = os.environ['MASTER_ADDR']
    else:
      raise ValueError(f"'{self.__class__.__name__}': missing master address "
                       "for rpc communication, try to provide it or set it "
                       "with environment variable 'MASTER_ADDR'")

    if master_port is not None:
      self.master_port = int(master_port)
    elif os.environ.get('MASTER_PORT') is not None:
      self.master_port = int(os.environ['MASTER_PORT']) + 1
    else:
      raise ValueError(f"'{self.__class__.__name__}': missing master port "
                       "for rpc communication, try to provide it or set it "
                       "with environment variable 'MASTER_ADDR'")

    self.num_rpc_threads = num_rpc_threads
    if self.num_rpc_threads is not None:
      assert self.num_rpc_threads > 0
    self.rpc_timeout = rpc_timeout

  def _set_worker_ranks(self, current_ctx: DistContext):
    self.worker_world_size = current_ctx.world_size * self.num_workers
    self.worker_ranks = [
      current_ctx.rank * self.num_workers + i
      for i in range(self.num_workers)
    ]

  def _assign_worker_devices(self):
    if self.worker_devices is not None:
      return
    self.worker_devices = [assign_device() for _ in range(self.num_workers)]


class CollocatedDistSamplingWorkerOptions(_BasicDistSamplingWorkerOptions):
  r""" Options for launching a single distributed sampling worker collocated
  with the current process.

  Args:
    master_addr (str, optional): Master address for rpc initialization across
      all sampling workers. (default: ``None``).
    master_port (str or int, optional): Master port for rpc initialization
      across all sampling workers. (default: ``None``).
    num_rpc_threads (int, optional): Number of threads used for rpc agent on
      each sampling worker. (default: ``None``).
    rpc_timeout (float): The timeout in seconds for rpc requests.
      (default: ``180``).

  Please ref to ``_BasicDistSamplingWorkerOptions`` for more detailed comments
  of related input arguments.
  """
  def __init__(self,
               master_addr: Optional[str] = None,
               master_port: Optional[Union[str, int]] = None,
               num_rpc_threads: Optional[int] = None,
               rpc_timeout: float = 180):
    super().__init__(1, None, 1, master_addr, master_port,
                     num_rpc_threads, rpc_timeout)


class MpDistSamplingWorkerOptions(_BasicDistSamplingWorkerOptions):
  r""" Options for launching distributed sampling workers with multiprocessing.

  Note that if ``MpDistWorkerOptions`` is used, all sampling workers will be
  launched on spawned subprocesses by ``torch.multiprocessing``. Thus, a
  share-memory based channel should be created for message passing of sampled
  results, which are produced by those multiprocessing sampling workers and
  consumed by the current process.

  Args:
    num_workers (int): How many workers to use (subprocesses to spwan) for
      distributed neighbor sampling of the current process. (default: ``1``).
    worker_devices (torch.device or List[torch.device], optional): List of
      devices assgined to workers of this group. (default: ``None``).
    worker_concurrency (int): The max sampling concurrency for each sampling
      worker. (default: ``4``).
    master_addr (str, optional): Master address for rpc initialization across
      all sampling workers. (default: ``None``).
    master_port (str or int, optional): Master port for rpc initialization
      across all sampling workers. (default: ``None``).
    num_rpc_threads (int, optional): Number of threads used for rpc agent on
      each sampling worker. (default: ``None``).
    rpc_timeout (float): The timeout in seconds for rpc requests.
      (default: ``180``).
    channel_size (int or str): The shared-memory buffer size (bytes) allocated
      for the channel. The number of ``num_workers * 64MB`` will be used if set
      to ``None``. (default: ``None``).
    pin_memory (bool): Set to ``True`` to register the underlying shared memory
      for cuda, which will achieve better performance if you want to copy
      loaded data from channel to cuda device. (default: ``False``).

  Please ref to ``_BasicDistSamplingWorkerOptions`` for more detailed comments
  of related input arguments.
  """
  def __init__(self,
               num_workers: int = 1,
               worker_devices: Optional[List[torch.device]] = None,
               worker_concurrency: int = 4,
               master_addr: Optional[str] = None,
               master_port: Optional[Union[str, int]] = None,
               num_rpc_threads: Optional[int] = None,
               rpc_timeout: float = 180,
               channel_size: Optional[Union[int, str]] = None,
               pin_memory: bool = False):
    super().__init__(num_workers, worker_devices, worker_concurrency,
                     master_addr, master_port, num_rpc_threads, rpc_timeout)

    self.channel_capacity = self.num_workers * self.worker_concurrency

    if channel_size is None:
      self.channel_size = f'{self.num_workers * 64}MB'
    else:
      self.channel_size = channel_size

    self.pin_memory = pin_memory


class RemoteDistSamplingWorkerOptions(_BasicDistSamplingWorkerOptions):
  r""" Options for launching distributed sampling workers on remote servers.

  Note that if ``RemoteDistSamplingWorkerOptions`` is used, all sampling
  workers will be launched on remote servers. Thus, a cross-machine based
  channel will be created for message passing of sampled results, which are
  produced by those remote sampling workers and consumed by the current process.

  Args:
    server_rank (int or List[int], optional): The rank of server to launch
      sampling workers, can be multiple. If set to ``None``, it will be 
      automatically assigned. (default: ``None``).
    num_workers (int): How many workers to launch on the remote server for
      distributed neighbor sampling of the current process. (default: ``1``).
    worker_devices (torch.device or List[torch.device], optional): List of
      devices assgined to workers of this group. (default: ``None``).
    worker_concurrency (int): The max sampling concurrency for each sampling
      worker. (default: ``4``).
    master_addr (str, optional): Master address for rpc initialization across
      all sampling workers. (default: ``None``).
    master_port (str or int, optional): Master port for rpc initialization
      across all sampling workers. (default: ``None``).
    num_rpc_threads (int, optional): Number of threads used for rpc agent on
      each sampling worker. (default: ``None``).
    rpc_timeout (float): The timeout in seconds for rpc requests.
      (default: ``180``).
    buffer_size (int or str): The size (bytes) allocated for the server-side
      buffer. The number of ``num_workers * 64MB`` will be used if set to
      ``None``. (default: ``None``).
    prefetch_size (int): The max prefetched sampled messages for consuming on
      the client side. (default: ``4``).
    glt_graph: Used in GraphScope side to get parameters. (default: ``None``).
    workload_type: Used in GraphScope side, indicates the type of option. This 
      field must be set when ``workload_type`` is not None. (default: ``None``).
  """
  def __init__(self,
               server_rank: Optional[Union[int, List[int]]] = None,
               num_workers: int = 1,
               worker_devices: Optional[List[torch.device]] = None,
               worker_concurrency: int = 4,
               master_addr: Optional[str] = None,
               master_port: Optional[Union[str, int]] = None,
               num_rpc_threads: Optional[int] = None,
               rpc_timeout: float = 180,
               buffer_size: Optional[Union[int, str]] = None,
               prefetch_size: int = 4,
               worker_key: str = None,
               glt_graph = None,
               workload_type: Optional[Literal['train', 'validate', 'test']] = None):
    # glt_graph is used in GraphScope side to get parameters
    if glt_graph:
      if not workload_type:
        raise ValueError(f"'{self.__class__.__name__}': missing workload_type ")
      master_addr = glt_graph.master_addr
      if workload_type == 'train':
        master_port = glt_graph.train_loader_master_port
      elif workload_type == 'validate':
        master_port = glt_graph.val_loader_master_port
      elif workload_type == 'test':
        master_port = glt_graph.test_loader_master_port
      worker_key = str(master_port)
    
    super().__init__(num_workers, worker_devices, worker_concurrency,
                     master_addr, master_port, num_rpc_threads, rpc_timeout)
    if server_rank is not None:
      self.server_rank = server_rank
    else:
      self.server_rank = assign_server_by_order()
    self.buffer_capacity = self.num_workers * self.worker_concurrency
    if buffer_size is None:
      self.buffer_size = f'{self.num_workers * 64}MB'
    else:
      self.buffer_size = buffer_size

    self.prefetch_size = prefetch_size
    if self.prefetch_size > self.buffer_capacity:
      raise ValueError(f"'{self.__class__.__name__}': the prefetch count "
                       f"{self.prefetch_size} exceeds the buffer capacity "
                       f"{self.buffer_capacity}")
    self.worker_key = worker_key


AllDistSamplingWorkerOptions = Union[
  CollocatedDistSamplingWorkerOptions,
  MpDistSamplingWorkerOptions,
  RemoteDistSamplingWorkerOptions
]
