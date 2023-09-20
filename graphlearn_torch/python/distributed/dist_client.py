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
from typing import Optional

from .dist_context import DistRole, get_context, _set_client_context
from .dist_server import DistServer, _call_func_on_server
from .rpc import init_rpc, shutdown_rpc, rpc_global_request_async, barrier


def init_client(num_servers: int, num_clients: int, client_rank: int,
                master_addr: str, master_port: int, num_rpc_threads: int = 4,
                client_group_name: Optional[str] = None, is_dynamic: bool = False):
  r""" Initialize the current process as a client and establish connections
  with all other servers and clients. Note that this method should be called
  only in the server-client distribution mode.

  Args:
    num_servers (int): Number of processes participating in the server group.
    num_clients (int): Number of processes participating in the client group.
    client_rank (int): Rank of the current process withing the client group (it
      should be a number between 0 and ``num_clients``-1).
    master_addr (str): The master TCP address for RPC connection between all
      servers and clients, the value of this parameter should be same for all
      servers and clients.
    master_port (int): The master TCP port for RPC connection between all
      servers and clients, the value of this parameter should be same for all
      servers and clients.
    num_rpc_threads (int): The number of RPC worker threads used for the
      current client. (Default: ``4``).
    client_group_name (str): A unique name of the client group that current
      process belongs to. If set to ``None``, a default name will be used.
      (Default: ``None``).
    is_dynamic (bool): Whether the world size is dynamic. (Default: ``False``).
  """
  if client_group_name:
    client_group_name = client_group_name.replace('-', '_')
  _set_client_context(num_servers, num_clients, client_rank, client_group_name)
  # Note that a client RPC agent will never remote requests, thus set the
  # number of rpc threads to ``1`` is enough.
  init_rpc(master_addr, master_port, num_rpc_threads=num_rpc_threads, is_dynamic=is_dynamic)


def shutdown_client():
  r""" Shutdown the client on the current process, notify other servers to
  exit, and destroy all connections.
  """
  current_context = get_context()
  if current_context is None:
    logging.warning("'shutdown_client': try to shutdown client when the "
                    "current process has not been initialized as a client.")
    return
  if not current_context.is_client():
    raise RuntimeError(f"'shutdown_client': role type of the current process "
                       f"context is not a client, got {current_context.role}.")
  # step 1: synchronize with all other clients.
  barrier()
  # step 2: use client-0 to notify all servers to exit after all clients
  # have reached here.
  current_context = get_context()
  if current_context.rank == 0:
    for server_rank in range(current_context.num_servers()):
      exit_status = request_server(server_rank, DistServer.exit)
      assert exit_status is True, f"Failed to exit server {server_rank}"
  # step 3: shutdown rpc across all servers and clients.
  shutdown_rpc()


def async_request_server(server_rank: int, func, *args, **kwargs):
  r""" The entry to perform an asynchronous request on a remote server, calling
  on the client side.
  """
  args = [func] + list(args)
  return rpc_global_request_async(
    target_role=DistRole.SERVER,
    role_rank=server_rank,
    func=_call_func_on_server,
    args=args,
    kwargs=kwargs,
  )


def request_server(server_rank: int, func, *args, **kwargs):
  r""" The entry to perform a synchronous request on a remote server, calling
  on the client side.
  """
  fut = async_request_server(server_rank, func, *args, **kwargs)
  return fut.wait()
