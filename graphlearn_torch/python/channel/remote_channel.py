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


class RemoteReceivingChannel(ChannelBase):
  r""" A pull-based receiving channel that can fetch sampled messages
  from remote sampling servers.

  Args:
    server_rank (int): The rank of target server to fetch sampled messages.
    producer_id (int): The sequence id of created sampling producer on the
      target server.
    num_expected (int): The number of expected sampled messages at one epoch.
    prefetch_size (int): The number of messages to prefetch. (Default ``4``).
  """
  def __init__(self,
               server_rank: int,
               producer_id: int,
               num_expected: int,
               prefetch_size: int = 4):
    self.server_rank = server_rank
    self.producer_id = producer_id
    self.num_expected = num_expected
    self.prefetch_size = prefetch_size
    self.num_request = 0
    self.num_received = 0
    self.queue = queue.Queue(maxsize=self.prefetch_size)

  def reset(self):
    r""" Reset all states to start a new epoch consuming.
    """
    # Discard messages that have not been consumed.
    while not self.queue.empty():
      _ = self.queue.get()
    self.num_request = 0
    self.num_received = 0

  def send(self, msg: SampleMessage, **kwargs):
    raise RuntimeError(f"'{self.__class__.__name__}': cannot send "
                       f"message with a receiving channel.")

  def recv(self, **kwargs) -> SampleMessage:
    self._request_some()
    msg = self.queue.get()
    self.num_received += 1
    return msg

  def _request_some(self):
    def on_done(f: torch.futures.Future):
      try:
        msg = f.wait()
        self.queue.put(msg)
      except Exception as e:
        logging.error("broken future of receiving remote messages: %s", e)

    from ..distributed import async_request_server, DistServer

    nun_req_limit = min(self.num_received + self.prefetch_size,
                        self.num_expected)
    for _ in range(nun_req_limit - self.num_request):
      fut = async_request_server(
        self.server_rank,
        DistServer.fetch_one_sampled_message,
        self.producer_id
      )
      fut.add_done_callback(on_done)
      self.num_request += 1
