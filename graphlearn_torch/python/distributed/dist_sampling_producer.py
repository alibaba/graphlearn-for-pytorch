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

import queue
import time
from enum import Enum
from typing import Optional, Union

import torch
import torch.multiprocessing as mp
from torch._C import _set_worker_signal_handlers
from torch.utils.data.dataloader import DataLoader

from ..channel import ChannelBase
from ..sampler import (
  NodeSamplerInput, EdgeSamplerInput, SamplingType, SamplingConfig
)
from ..utils import ensure_device

from ..distributed.dist_context import get_context
from .dist_context import init_worker_group
from .dist_dataset import DistDataset
from .dist_neighbor_sampler import DistNeighborSampler
from .dist_options import _BasicDistSamplingWorkerOptions
from .rpc import init_rpc, shutdown_rpc


MP_STATUS_CHECK_INTERVAL = 5.0
r""" Interval (in seconds) to check status of processes to avoid hanging in
multiprocessing sampling.
"""


class MpCommand(Enum):
  r""" Enum class for multiprocessing sampling command
  """
  SAMPLE_ALL = 0
  STOP = 1


def _sampling_worker_loop(rank,
                          data: DistDataset,
                          sampler_input: Union[NodeSamplerInput, EdgeSamplerInput],
                          unshuffled_index: Optional[torch.Tensor],
                          sampling_config: SamplingConfig,
                          worker_options: _BasicDistSamplingWorkerOptions,
                          channel: ChannelBase,
                          task_queue: mp.Queue,
                          sampling_completed_worker_count: mp.Value,
                          mp_barrier):
  r""" Subprocess work loop for sampling worker.
  """
  dist_sampler = None
  try:
    init_worker_group(
      world_size=worker_options.worker_world_size,
      rank=worker_options.worker_ranks[rank],
      group_name='_sampling_worker_subprocess'
    )

    if worker_options.num_rpc_threads is None:
      num_rpc_threads = min(data.num_partitions, 16)
    else:
      num_rpc_threads = worker_options.num_rpc_threads

    current_device = worker_options.worker_devices[rank]
    ensure_device(current_device)

    _set_worker_signal_handlers()
    torch.set_num_threads(num_rpc_threads + 1)

    init_rpc(
      master_addr=worker_options.master_addr,
      master_port=worker_options.master_port,
      num_rpc_threads=num_rpc_threads,
      rpc_timeout=worker_options.rpc_timeout
    )

    dist_sampler = DistNeighborSampler(
      data, sampling_config.num_neighbors, sampling_config.with_edge,
      sampling_config.with_neg, sampling_config.with_weight,
      sampling_config.edge_dir, sampling_config.collect_features, channel,
      worker_options.worker_concurrency, current_device
    )
    dist_sampler.start_loop()

    if unshuffled_index is not None:
      unshuffled_index_loader = DataLoader(
        unshuffled_index, batch_size=sampling_config.batch_size,
        shuffle=False, drop_last=sampling_config.drop_last
      )
    else:
      unshuffled_index_loader = None

    mp_barrier.wait()

    keep_running = True
    while keep_running:
      try:
        command, args = task_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
      except queue.Empty:
        continue

      if command == MpCommand.SAMPLE_ALL:
        seeds_index = args
        if seeds_index is None:
          loader = unshuffled_index_loader
        else:
          loader = DataLoader(
            seeds_index, batch_size=sampling_config.batch_size,
            shuffle=False, drop_last=sampling_config.drop_last
          )

        if sampling_config.sampling_type == SamplingType.NODE:
          for index in loader:
            dist_sampler.sample_from_nodes(sampler_input[index])
        elif sampling_config.sampling_type == SamplingType.LINK:
          for index in loader:
            dist_sampler.sample_from_edges(sampler_input[index])
        elif sampling_config.sampling_type == SamplingType.SUBGRAPH:
          for index in loader:
            dist_sampler.subgraph(sampler_input[index])

        dist_sampler.wait_all()

        with sampling_completed_worker_count.get_lock():
          sampling_completed_worker_count.value += 1 # non-atomic, lock is necessary

      elif command == MpCommand.STOP:
        keep_running = False
      else:
        raise RuntimeError("Unknown command type")
  except KeyboardInterrupt:
    # Main process will raise KeyboardInterrupt anyways.
    pass

  if dist_sampler is not None:
    dist_sampler.shutdown_loop()
  shutdown_rpc(graceful=False)


class DistMpSamplingProducer(object):
  r""" A subprocess group of distributed sampling workers.

  Note that this producer is only used for workload with separate sampling
  and training, all sampled results will be sent to the output channel.
  """
  def __init__(self,
               data: DistDataset,
               sampler_input: Union[NodeSamplerInput, EdgeSamplerInput],
               sampling_config: SamplingConfig,
               worker_options: _BasicDistSamplingWorkerOptions,
               output_channel: ChannelBase):
    self.data = data
    self.sampler_input = sampler_input.share_memory()
    self.input_len = len(self.sampler_input)
    self.sampling_config = sampling_config
    self.worker_options = worker_options
    self.worker_options._assign_worker_devices()
    self.num_workers = self.worker_options.num_workers
    self.output_channel = output_channel
    self.sampling_completed_worker_count = mp.Value('I', lock=True)
    current_ctx = get_context()
    self.worker_options._set_worker_ranks(current_ctx)  

    self._task_queues = []
    self._workers = []
    self._barrier = None
    self._shutdown = False
    self._worker_seeds_ranges = self._get_worker_seeds_ranges()

  def init(self):
    r""" Create the subprocess pool. Init samplers and rpc server.
    """
    if not self.sampling_config.shuffle:
      unshuffled_indexes = self._get_seeds_indexes()
    else:
      unshuffled_indexes = [None] * self.num_workers

    mp_context = mp.get_context('spawn')
    barrier = mp_context.Barrier(self.num_workers + 1)
    for rank in range(self.num_workers):
      task_queue = mp_context.Queue(
        self.num_workers * self.worker_options.worker_concurrency)
      self._task_queues.append(task_queue)
      w = mp_context.Process(
        target=_sampling_worker_loop,
        args=(rank, self.data, self.sampler_input, unshuffled_indexes[rank],
              self.sampling_config, self.worker_options, self.output_channel,
              task_queue, self.sampling_completed_worker_count, barrier)
      )
      w.daemon = True
      w.start()
      self._workers.append(w)
    barrier.wait()

  def shutdown(self):
    r""" Shutdown sampler event loop and rpc server. Join the subprocesses.
    """
    if not self._shutdown:
      self._shutdown = True
      try:
        for q in self._task_queues:
          q.put((MpCommand.STOP, None))
        for w in self._workers:
          w.join(timeout=MP_STATUS_CHECK_INTERVAL)
        for q in self._task_queues:
          q.cancel_join_thread()
          q.close()
      finally:
        for w in self._workers:
          if w.is_alive():
            w.terminate()

  def produce_all(self):
    r""" Perform sampling for all input seeds.
    """
    if self.sampling_config.shuffle:
      seeds_indexes = self._get_seeds_indexes()
      for rank in range(self.num_workers):
        seeds_indexes[rank].share_memory_()
    else:
      seeds_indexes = [None] * self.num_workers

    self.sampling_completed_worker_count.value = 0
    for rank in range(self.num_workers):
      self._task_queues[rank].put((MpCommand.SAMPLE_ALL, seeds_indexes[rank]))
    time.sleep(0.1)

  def is_all_sampling_completed_and_consumed(self):
    if self.output_channel.empty():
      return self.is_all_sampling_completed()
 
  def is_all_sampling_completed(self):
    return self.sampling_completed_worker_count.value == self.num_workers

  def _get_worker_seeds_ranges(self):
    num_worker_batches = [0] * self.num_workers
    num_total_complete_batches = (self.input_len // self.sampling_config.batch_size)
    for rank in range(self.num_workers):
      num_worker_batches[rank] += \
        (num_total_complete_batches // self.num_workers)
    for rank in range(num_total_complete_batches % self.num_workers):
      num_worker_batches[rank] += 1

    index_ranges = []
    start = 0
    for rank in range(self.num_workers):
      end = start + num_worker_batches[rank] * self.sampling_config.batch_size
      if rank == self.num_workers - 1:
        end = self.input_len
      index_ranges.append((start, end))
      start = end

    return index_ranges

  def _get_seeds_indexes(self):
    if self.sampling_config.shuffle:
      index = torch.randperm(self.input_len)
    else:
      index = torch.arange(self.input_len)

    seeds_indexes = []
    for rank in range(self.num_workers):
      start, end = self._worker_seeds_ranges[rank]
      seeds_indexes.append(index[start:end])

    return seeds_indexes


class DistCollocatedSamplingProducer(object):
  r""" A sampling producer with a collocated distributed sampler.

  Note that the sampled results will be returned directly and this producer
  will be blocking when processing each batch.
  """
  def __init__(self,
               data: DistDataset,
               sampler_input: Union[NodeSamplerInput, EdgeSamplerInput],
               sampling_config: SamplingConfig,
               worker_options: _BasicDistSamplingWorkerOptions,
               device: torch.device):
    self.data = data
    self.sampler_input = sampler_input
    self.sampling_config = sampling_config
    self.worker_options = worker_options
    self.device = device

  def init(self):
    index = torch.arange(len(self.sampler_input))

    self._index_loader = DataLoader(
      index,
      batch_size=self.sampling_config.batch_size,
      shuffle=self.sampling_config.shuffle,
      drop_last=self.sampling_config.drop_last
    )
    self._index_iter = self._index_loader._get_iterator()

    if self.worker_options.num_rpc_threads is None:
      num_rpc_threads = min(self.data.num_partitions, 16)
    else:
      num_rpc_threads = self.worker_options.num_rpc_threads

    init_rpc(
      master_addr=self.worker_options.master_addr,
      master_port=self.worker_options.master_port,
      num_rpc_threads=num_rpc_threads,
      rpc_timeout=self.worker_options.rpc_timeout
    )

    self._collocated_sampler = DistNeighborSampler(
      self.data, self.sampling_config.num_neighbors,
      self.sampling_config.with_edge, self.sampling_config.with_neg,
      self.sampling_config.with_weight,
      self.sampling_config.edge_dir, self.sampling_config.collect_features,
      channel=None, concurrency=1, device=self.device
    )
    self._collocated_sampler.start_loop()

  def shutdown(self):
    if self._collocated_sampler is not None:
      self._collocated_sampler.shutdown_loop()

  def reset(self):
    self._index_iter._reset(self._index_loader)

  def sample(self):
    index = self._index_iter._next_data()
    batch = self.sampler_input[index]
    if self.sampling_config.sampling_type == SamplingType.NODE:
      return self._collocated_sampler.sample_from_nodes(batch)
    if self.sampling_config.sampling_type == SamplingType.LINK:
      return self._collocated_sampler.sample_from_edges(batch)
    if self.sampling_config.sampling_type == SamplingType.SUBGRAPH:
      return self._collocated_sampler.subgraph(batch)
    raise NotImplementedError
