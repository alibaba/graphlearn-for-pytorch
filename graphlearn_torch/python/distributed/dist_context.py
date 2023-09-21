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

from enum import Enum
from typing import Optional, List


class DistRole(Enum):
  r""" Role types for distributed context groups.
  """
  WORKER = 1  # As a worker in a distributed worker group (non-server mode)
  SERVER = 2  # As a server in a distributed server group (server-client mode)
  CLIENT = 3  # As a client in a distributed client group (server-client mode)


_DEFAULT_WORKER_GROUP = '_default_worker'
_DEFAULT_SERVER_GROUP = '_default_server'
_DEFAULT_CLIENT_GROUP = '_default_client'


class DistContext(object):
  r""" Distributed context info of the current process.

  Args:
    role (DistRole): The role type of the current context group.
    group_name (str): A unique name of the current role group.
    world_size (int): The number of processes in the current role group.
    rank (int): The current process rank within the current role group.
    global_world_size (int): The total number of processes in all role groups.
    global_rank (int): The current process rank within all role groups.
  """
  def __init__(self,
               role: DistRole,
               group_name: str,
               world_size: int,
               rank: int,
               global_world_size: int,
               global_rank: int):
    assert world_size > 0 and rank in range(world_size)
    assert global_world_size > 0 and global_rank in range(global_world_size)
    assert world_size <= global_world_size
    self.role = role
    self.group_name = group_name
    self.world_size = world_size
    self.rank = rank
    self.global_world_size = global_world_size
    self.global_rank = global_rank

  def __repr__(self) -> str:
    cls = self.__class__.__name__
    info = []
    for key, value in self.__dict__.items():
      info.append(f"{key}: {value}")
    info = ", ".join(info)
    return f"{cls}({info})"

  def __eq__(self, obj):
    if not isinstance(obj, DistContext):
      return False
    for key, value in self.__dict__.items():
      if value != obj.__dict__[key]:
        return False
    return True

  def is_worker(self) -> bool:
    return self.role == DistRole.WORKER

  def is_server(self) -> bool:
    return self.role == DistRole.SERVER

  def is_client(self) -> bool:
    return self.role == DistRole.CLIENT

  def num_servers(self) -> int:
    if self.role == DistRole.SERVER:
      return self.world_size
    if self.role == DistRole.CLIENT:
      return self.global_world_size - self.world_size
    return 0

  def num_clients(self) -> int:
    if self.role == DistRole.CLIENT:
      return self.world_size
    if self.role == DistRole.SERVER:
      return self.global_world_size - self.world_size
    return 0

  @property
  def worker_name(self) -> str:
    r""" Get worker name of the current process of this context.
    """
    return f"{self.group_name}_{self.rank}"


_dist_context: DistContext = None
r""" Distributed context on the current process.
"""
_clients_to_servers: dict = None
r""" A dict mapping from client rank to server ranks. int -> List[int]"""

def get_context() -> DistContext:
  r""" Get distributed context info of the current process.
  """
  return _dist_context

def get_clients_to_servers() -> dict:
  r""" Get client to servers mapping.
  """
  return _clients_to_servers

def _set_worker_context(world_size: int, rank: int,
                        group_name: Optional[str] = None):
  r""" Set distributed context info as a non-server worker on the current
  process.
  """
  global _dist_context
  _dist_context = DistContext(
    role=DistRole.WORKER,
    group_name=(group_name if group_name is not None
                else _DEFAULT_WORKER_GROUP),
    world_size=world_size,
    rank=rank,
    global_world_size=world_size,
    global_rank=rank
  )


def _set_server_context(num_servers: int, server_rank: int,
                        server_group_name: Optional[str] = None, num_clients: int = 0):
  r""" Set distributed context info as a server on the current process.
  """
  assert num_servers > 0
  global _dist_context
  _dist_context = DistContext(
    role=DistRole.SERVER,
    group_name=(server_group_name if server_group_name is not None
                else _DEFAULT_SERVER_GROUP),
    world_size=num_servers,
    rank=server_rank,
    global_world_size=num_servers+num_clients,
    global_rank=server_rank
  )


def _set_client_context(num_servers: int, num_clients: int, client_rank: int,
                        client_group_name: Optional[str] = None):
  r""" Set distributed context info as a client on the current process.
  """
  assert num_servers > 0 and num_clients > 0
  global _dist_context
  _dist_context = DistContext(
    role=DistRole.CLIENT,
    group_name=(client_group_name if client_group_name is not None
                else _DEFAULT_CLIENT_GROUP),
    world_size=num_clients,
    rank=client_rank,
    global_world_size=num_servers+num_clients,
    global_rank=num_servers+client_rank
  )
  assign_server_by_order()

def assign_server_by_order():
  r"""Assign servers to each client in turn.
  e.g. 2 clients and 4 servers, then the assignment is: {0: [0, 1], 1: [2, 3]},
  5 clients and 2 servers, then the assignment is: {0: [0], 1: [1], 2: [0], 3: [1], 4: [0]}."""
  ctx = get_context()
  assert ctx is not None and ctx.is_client()
  client_num, server_num = ctx.world_size, ctx.global_world_size - ctx.world_size
  global _clients_to_servers
  _clients_to_servers = {}
  cur_server = 0
  for i in range(client_num):
    if i not in _clients_to_servers:
      _clients_to_servers[i] = []
    for j in range(server_num // client_num):
      _clients_to_servers[i].append(cur_server)
      cur_server  = (cur_server + 1) % server_num
    if i < server_num % client_num:
      _clients_to_servers[i].append(cur_server)
      cur_server = (cur_server + 1) % server_num
    if len(_clients_to_servers[i]) == 0:
      _clients_to_servers[i].append(cur_server)
      cur_server = (cur_server + 1) % server_num
  return _clients_to_servers[ctx.rank]


def init_worker_group(world_size: int, rank: int,
                      group_name: Optional[str] = None):
  r""" Initialize a simple worker group on the current process, this method
  should be called only in a non-server distribution mode with a group of
  parallel workers.

  Args:
    world_size (int): Number of all processes participating in the distributed
      worker group.
    rank (int): Rank of the current process withing the distributed group (it
      should be a number between 0 and ``world_size``-1).
    group_name (str): A unique name of the distributed group that current
      process belongs to. If set to ``None``, a default name will be used.
  """
  _set_worker_context(world_size, rank, group_name)