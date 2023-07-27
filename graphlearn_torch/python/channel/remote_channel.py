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
import queue
import torch

from .base import SampleMessage, ChannelBase
from typing import Union, List


class RemoteReceivingChannel(ChannelBase):
  r""" A pull-based receiving channel that can fetch sampled messages
  from remote sampling servers.

  Args:
    server_rank (int or List[int]): The ranks of target server to fetch sampled 
      messages.
    producer_id (int or List[int]) ): The sequence ids of created sampling producer 
      on the target server.
    prefetch_size (int): The number of messages to prefetch for every server. 
      (Default ``2``).
  """

  def __init__(
    self,
    server_rank: Union[int, List[int]],
    producer_id: Union[int, List[int]],
    prefetch_size: int = 2
  ):
    self.server_rank_list = server_rank if isinstance(server_rank,
                                                      List) else [server_rank]
    self.producer_id_list = producer_id if isinstance(producer_id,
                                                      List) else [producer_id]
    self.prefetch_size = prefetch_size

    assert len(self.server_rank_list) == len(self.producer_id_list)

    self.num_request_list = [0] * len(self.server_rank_list)
    self.num_received_list = [0] * len(self.server_rank_list)
    self.server_end_of_epoch = [False] * len(self.server_rank_list)
    self.global_end_of_epoch = False

    self.queue = queue.Queue(maxsize=self.prefetch_size * len(self.server_rank_list))

  def reset(self):
    r""" Reset all states to start a new epoch consuming.
    """
    # Discard messages that have not been consumed.
    while not self.queue.empty():
      _ = self.queue.get()
  
    self.server_end_of_epoch = [False] * len(self.server_rank_list)
    self.num_request_list = [0] * len(self.server_rank_list)
    self.num_received_list = [0] * len(self.server_rank_list)
    self.global_end_of_epoch = False

  def send(self, msg: SampleMessage, **kwargs):
    raise RuntimeError(
      f"'{self.__class__.__name__}': cannot send "
      f"message with a receiving channel."
    )

  def recv(self, **kwargs) -> SampleMessage:
    if self.global_end_of_epoch:
      if self._all_received():
        raise StopIteration
    else:
      self._request_some()
    msg, end_of_epoch, local_server_idx = self.queue.get()
    self.num_received_list[local_server_idx] += 1
    
    # server guarantees that when end_of_epoch is true, msg must be None
    while end_of_epoch:
      self.server_end_of_epoch[local_server_idx] = True
      if sum(self.server_end_of_epoch) == len(self.server_rank_list):
        self.global_end_of_epoch = True
        if self._all_received():
          raise StopIteration
      msg, end_of_epoch, local_server_idx = self.queue.get()
      self.num_received_list[local_server_idx] += 1
    return msg
  
  def _all_received(self):
    return sum(self.num_received_list) == sum(self.num_request_list)

  def _request_some(self):

    def on_done(f: torch.futures.Future, local_server_idx):
      try:
        msg, end_of_epoch = f.wait()
        self.queue.put((msg, end_of_epoch, local_server_idx))

      except Exception as e:
        logging.error("broken future of receiving remote messages: %s", e)

    def create_callback(local_server_idx):

      def callback(f):
        on_done(f, local_server_idx)

      return callback

    from ..distributed import async_request_server, DistServer

    for local_server_idx, server_rank in enumerate(self.server_rank_list):
      if not self.server_end_of_epoch[local_server_idx]:
        for _ in range(
          self.num_received_list[local_server_idx] +
          self.prefetch_size -
          self.num_request_list[local_server_idx]
        ):
          fut = async_request_server(
            server_rank, DistServer.fetch_one_sampled_message,
            self.producer_id_list[local_server_idx]
          )
          cb = create_callback(local_server_idx)
          fut.add_done_callback(cb)
          self.num_request_list[local_server_idx] += 1
