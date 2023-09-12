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

import atexit
import time
import collections
import functools
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Set

from torch.distributed import rpc

from .dist_context import DistRole, get_context

SERVER_INIT_CHECK_INTERVAL = 3.0
MAX_RETYR_TIMES = 60

_rpc_init_lock = threading.RLock()

_rpc_inited: bool = False
r""" State of rpc initialization on the current process.
"""

_rpc_worker_names: Dict[DistRole, List[str]] = None
r""" Dict from role type to all rpc worker names in this role group.
"""

_rpc_current_group_worker_names: Set[str] = None
r""" Set of rpc worker names in the current role group. Used in all_gather
in a role group. 
"""

_rpc_master_addr: str = None
r""" Master address used by rpc agent on the current process.
"""

_rpc_master_port: int = None
r""" Master port used by rpc agent on the current process.
"""


def rpc_is_initialized():
  r""" Check whether rpc has been initialized on the current process.
  """
  return _rpc_inited


def _require_initialized(func):
  r""" A function wrapper to check whether RPC has been initialized, otherwise
  an error will be raised. Note that the implementation of this method is same
  to ``torch.distributed.rpc.api._require_initialized``.
  """
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    if rpc_is_initialized() is not True:
      raise RuntimeError( "RPC has not been initialized or has been shutdowned")
    return func(*args, **kwargs)

  return wrapper


@_require_initialized
def get_rpc_master_addr():
  r""" Get the master address for rpc communication on the current process.
  """
  return _rpc_master_addr


@_require_initialized
def get_rpc_master_port():
  r""" Get the master port for rpc communication on the current process.
  """
  return _rpc_master_port

@_require_initialized
def get_rpc_current_group_worker_names() -> List[str]:
  r""" Get the rpc worker names (sorted by rank) of the current group.
  """
  current_role = get_context().role
  return _rpc_worker_names[current_role]

@_require_initialized
def get_rpc_worker_names() -> Dict[DistRole, List[str]]:
  r""" Get the rpc worker names (each sorted by rank) of each current group.
  """
  return _rpc_worker_names


## All gather objects only from the current role group.

_role_based_all_gather_dict_lock = threading.RLock()
_role_based_all_gather_sequence_id = 0
_role_based_all_gather_sequence_id_to_states: collections.defaultdict = \
  collections.defaultdict(rpc.AllGatherStates)


def _role_based_gather_to_leader(sequence_id, worker_name, obj):
  with _role_based_all_gather_dict_lock:
    assert (
      worker_name in _rpc_current_group_worker_names
    ), f"{worker_name} is not expected by leader."
    states = _role_based_all_gather_sequence_id_to_states[sequence_id]
    assert (
      worker_name not in states.gathered_objects
    ), f"{worker_name} reported intent sequence id {sequence_id} twice."
    states.gathered_objects[worker_name] = obj
    if _rpc_current_group_worker_names == set(states.gathered_objects.keys()):
      states.proceed_signal.set()


def _role_based_broadcast_to_followers(sequence_id, objects_map):
  with _role_based_all_gather_dict_lock:
    states = _role_based_all_gather_sequence_id_to_states[sequence_id]

    assert (
      not states.proceed_signal.is_set()
    ), f"Termination signal sequence id {sequence_id} got set twice."
    states.gathered_objects = objects_map
    states.proceed_signal.set()


@_require_initialized
def all_gather(obj, timeout=None):
  r""" Gathers objects only from the current role group in a list. This
  function blocks until all workers in the current role group have received
  the gathered results. The implementation of this method is refer to
  ``torch.distributed.rpc.api._all_gather``.
  """
  assert (
    _rpc_current_group_worker_names is not None
  ), "`_rpc_current_group_worker_names` is not initialized for `all_gather`."
  leader_name = sorted(_rpc_current_group_worker_names)[0]
  self_name = get_context().worker_name

  global _role_based_all_gather_sequence_id
  with _role_based_all_gather_dict_lock:
    sequence_id = _role_based_all_gather_sequence_id
    _role_based_all_gather_sequence_id += 1

  is_leader = leader_name == self_name
  if timeout is None:
    timeout = rpc.get_rpc_timeout()

  # Phase 1: Followers send it's object to the leader
  if is_leader:
    _role_based_gather_to_leader(sequence_id, self_name, obj)
  else:
    rpc.rpc_sync(
      leader_name,
      _role_based_gather_to_leader,
      args=(sequence_id, self_name, obj),
      timeout=timeout,
    )

  with _role_based_all_gather_dict_lock:
    states = _role_based_all_gather_sequence_id_to_states[sequence_id]
  states.proceed_signal.wait()

  # Phase 2: Leader broadcast gathered results to all followers
  # Leader's signal is the first to be unblocked, after receiving all
  # followers' data objects.
  if is_leader:
    worker_name_to_response_future_dict = {}
    for follower_name in _rpc_current_group_worker_names - {leader_name}:
      fut = rpc.rpc_async(
        follower_name,
        _role_based_broadcast_to_followers,
        args=(sequence_id, states.gathered_objects),
        timeout=timeout
      )
      worker_name_to_response_future_dict[follower_name] = fut

    errors = []
    for follower_name, fut in worker_name_to_response_future_dict.items():
      try:
        fut.wait()
      except RuntimeError as ex:
        errors.append((follower_name, ex))

    if errors:
      raise RuntimeError(
        f"Followers {[e[0] for e in errors]} timed out in all_gather "
        f"after {timeout:.2f} seconds. The first exception is {errors[0][1]}"
      )

  return states.gathered_objects


@_require_initialized
def barrier(timeout=None):
  r""" Block until all local and remote RPC processes in the current role
  group reach this method.
  """
  try:
    all_gather(obj=None, timeout=timeout)
  except RuntimeError as ex:
    logging.error("Failed to respond to 'barrier' in time, got error: %s", ex)


## All gather objects from all role groups.

@_require_initialized
def global_all_gather(obj, timeout=None):
  r""" Gathers objects from all role groups in a list, using the implementation
  of ``torch.distributed.rpc.api._all_gather``.
  """
  if timeout is None:
    return rpc.api._all_gather(obj)
  return rpc.api._all_gather(obj, timeout=timeout)


@_require_initialized
def global_barrier(timeout=None):
  r""" Block until all local and remote RPC processes across all role groups
  reach this method.
  """
  try:
    global_all_gather(obj=None, timeout=timeout)
  except RuntimeError as ex:
    logging.error("Failed to respond to 'global_barrier' "
                  "in time, got error %s", ex)


## RPC initialization and shutdown

def init_rpc(master_addr: str,
             master_port: int,
             num_rpc_threads: int = 16,
             rpc_timeout: float = 180,
             is_dynamic: bool = False):
  r""" Initialize rpc on the current process.
  """
  with _rpc_init_lock:
    if rpc_is_initialized() is True:
      return
    if rpc_is_initialized() is None:
      raise RuntimeError("'init_rpc': Try to re-init rpc after shutdown.")

    ctx = get_context()
    if ctx is None:
      raise RuntimeError("'init_rpc': Distributed context has not been set.")

    options = rpc.TensorPipeRpcBackendOptions(
      _transports=['ibv', 'uv'],
      _channels=['mpt_uv', 'basic'],
      num_worker_threads=num_rpc_threads,
      rpc_timeout=rpc_timeout,
      init_method=f'tcp://{master_addr}:{master_port}'
    )
    
    rpc.init_rpc(
      name=ctx.worker_name,
      rank=ctx.global_rank,
      world_size=None if is_dynamic else ctx.global_world_size,
      rpc_backend_options=options
    )

    global _rpc_inited
    _rpc_inited = True

    global _rpc_current_group_worker_names
    global _rpc_worker_names
    _rpc_worker_names = {}
    
    if is_dynamic:
      _rpc_worker_names[DistRole.SERVER] = []
      _rpc_worker_names[DistRole.CLIENT] = [] 
     
      if ctx.is_server():
        # ensure all servers is inited
        for server_rank in range(ctx.world_size):  
          if server_rank == ctx.rank:
            _rpc_worker_names[DistRole.SERVER].append(ctx.group_name + '_' + str(server_rank))
            continue
          times = 0
          is_avail = False
          while not is_avail:
            try:
              is_avail = rpc_global_request_by_rank(server_rank, rpc.is_available)
            except:
              time.sleep(SERVER_INIT_CHECK_INTERVAL)
              logging.info(f"RETRY {times}: server {ctx.rank} waits server {server_rank}...")
            times += 1
            if times >= MAX_RETYR_TIMES:
              raise RuntimeError(f"TIMEOUT: server {ctx.rank} waits server {server_rank} timeout."
                                 f"Check if server {server_rank} is ready.")
          _rpc_worker_names[DistRole.SERVER].append(ctx.group_name + '_' + str(server_rank))
        _rpc_current_group_worker_names = set(_rpc_worker_names[DistRole.SERVER])
        return
      if ctx.is_client():
        for server_rank in range(ctx.global_rank - ctx.rank):
          times = 0
          is_avail = False
          while not is_avail:
            try:
              is_avail = rpc_global_request_by_rank(server_rank, rpc.is_available)
            except:
              time.sleep(SERVER_INIT_CHECK_INTERVAL)
              logging.info(f"RETRY {times}: client {ctx.rank} waits server {server_rank}...")
            times += 1
            if times >= MAX_RETYR_TIMES:
              raise RuntimeError(f"TIMEOUT: client {ctx.rank} waits server {server_rank} timeout."
                                 f"Check if server {server_rank} is ready.")
          server_name = rpc_global_request_by_rank(server_rank, rpc.get_worker_info).name
          _rpc_worker_names[DistRole.SERVER].append(server_name)
        _rpc_current_group_worker_names = set([ctx.group_name + '_' + str(client_rank) for client_rank in range(ctx.world_size)])
        return
    
    gathered_results = global_all_gather(
      obj=(ctx.role, ctx.world_size, ctx.rank), timeout=rpc_timeout
    )
    for worker_name, (role, role_size, role_rank) in gathered_results.items():
      worker_list = _rpc_worker_names.get(role, None)
      if worker_list is None:
        worker_list = [None for _ in range(role_size)]
      else:
        if len(worker_list) != role_size:
          raise RuntimeError(f"'init_rpc': world size of role {role} gathered "
                             f"from {worker_name} is inconsistent with others.")
      if worker_list[role_rank] is not None:
        raise RuntimeError(f"'init_rpc': try to set worker name twice with "
                           f"the same rank {role_rank} of role {role}")
      worker_list[role_rank] = worker_name
      _rpc_worker_names[role] = worker_list

    _rpc_current_group_worker_names = set(_rpc_worker_names[ctx.role])

    global_barrier(timeout=rpc_timeout) 
    # TODO(hongyi): in server-client mode, if "torch.distributed.init_process_group" follows "global_barrier", 
    # some participants may randomly hang
    time.sleep(1) 

def shutdown_rpc(graceful=True):
  r""" Shutdown rpc agent on the current process.

  If `graceful` set to `False`, other mechanisms should ensure that all
  rpc requests are completed before shutting down rpc servers.
  """
  if rpc_is_initialized() is True:
    rpc.shutdown(graceful=graceful)
    global _rpc_inited
    _rpc_inited = None

atexit.register(shutdown_rpc, False)


## RPC synchronization and routing with data partition mapping.

class RpcDataPartitionRouter(object):
  r""" A router that can select a remote rpc worker with a certain data
  partition to perform a rpc request.
  """
  def __init__(self, partition2workers: List[List[str]]):
    for pidx, rpc_worker_list in enumerate(partition2workers):
      if len(rpc_worker_list) == 0:
        raise ValueError(f"'RpcDataPartitionRouter': no rpc worker is "
                         f"responsible for data partition '{pidx}'.")
    self.partition2workers = partition2workers
    self.rpc_worker_indexs = [0 for _ in range(len(partition2workers))]

  def get_to_worker(self, data_partition_idx: int) -> str:
    rpc_worker_list = self.partition2workers[data_partition_idx]
    worker_index = self.rpc_worker_indexs[data_partition_idx]
    to_worker = rpc_worker_list[worker_index]
    self.rpc_worker_indexs[data_partition_idx] = \
      (worker_index + 1) % len(rpc_worker_list)
    return to_worker


@_require_initialized
def rpc_sync_data_partitions(num_data_partitions: int,
                             current_partition_idx: int):
  r""" Synchronize the data partition info across all workers only in the
  current role group.

  Note that all data should be partitioned and used with a single role group.

  Args:
    num_data_partitions (int): The number of all data partitions.
    current_partition_idx (int): The data partition idx that the current
      process is responsible for, some compution tasks on this data partition
      may be send to the current process from remote workers.
  """
  ctx = get_context()
  partition2workers  = [[] for _ in range(num_data_partitions)]
  gathered_results = all_gather(
    (ctx.role, num_data_partitions, current_partition_idx))
  for worker_name, (role, nparts, idx) in gathered_results.items():
    if role != ctx.role:
      raise RuntimeError(f"'rpc_sync_data_partition_mapping': inconsistent "
                         f"role type '{role}' gathered from {worker_name}, "
                         f"current role type is '{ctx.role}'.")
    if nparts != num_data_partitions:
      raise RuntimeError(f"'rpc_sync_data_partition_mapping': inconsistent "
                         f"data partition number '{nparts}' gathered from "
                         f"{worker_name}, the value on current process is "
                         f"'{ctx.role}'.")
    partition2workers[idx].append(worker_name)
  return partition2workers


## RPC registration in the current role group.

class RpcCalleeBase(ABC):
  r""" A wrapper base for rpc callee that will perform rpc requests from
  remote processes.

  Note that the callee will be called only from rpc workers in the current
  role group.
  """
  def __init__(self):
    pass

  @abstractmethod
  def call(self, *args, **kwargs):
    r""" The real processing entry for rpc requests, need to be overwrite.
    """


_rpc_callee_lock = threading.RLock()
_rpc_callee_id: int = 0
_rpc_callee_pool: Dict[int, RpcCalleeBase] = {}


@_require_initialized
def rpc_register(callee: RpcCalleeBase):
  r""" Register a callee for rpc requests only in the current role group,
  this method will block until all local and remote RPC processes of the
  current role group reach this method.
  """
  global _rpc_callee_id, _rpc_callee_pool
  with _rpc_callee_lock:
    callee_id = _rpc_callee_id
    _rpc_callee_id += 1
    if callee_id in _rpc_callee_pool:
      raise RuntimeError(f"'rpc_register': try to register with the "
                         f"callee id {callee_id} twice.")
    _rpc_callee_pool[callee_id] = callee

  current_role = get_context().role
  callee_ids = all_gather((current_role, callee_id))
  for name, (role, cid) in callee_ids.items():
    if role != current_role:
      raise RuntimeError(f"'rpc_register': get inconsistent role '{role}' "
                         f"from {name}, current role is '{current_role}'.")
    if cid != callee_id:
      raise RuntimeError(f"'rpc_register': get inconsistent callee id '{cid}' "
                         f"from {name}, current callee id is '{callee_id}'.")

  return callee_id


## RPC request entries only for the current role group.

def _rpc_call(callee_id, *args, **kwargs):
  r""" Entry for rpc requests within the current role group.
  """
  return _rpc_callee_pool.get(callee_id).call(*args, **kwargs)


@_require_initialized
def rpc_request_async(worker_name, callee_id, args=None, kwargs=None):
  r""" Perform a rpc request asynchronously within the current role
  group. and return a future.
  """
  return rpc.rpc_async(
    to=worker_name,
    func=_rpc_call,
    args=(callee_id, *args),
    kwargs=kwargs
  )


@_require_initialized
def rpc_request(worker_name, callee_id, args=None, kwargs=None):
  r""" Perform a rpc request synchronously within the current role
  group and return the results.
  """
  fut = rpc_request_async(worker_name, callee_id, args, kwargs)
  return fut.wait()


## RPC request entries to other rpc worker on arbitrary role group.

@_require_initialized
def rpc_global_request_async(target_role: DistRole, role_rank: int,
                             func, args=None, kwargs=None):
  r""" Perform a rpc request asynchronously to other rpc worker on
  arbitrary role group and return a future.
  """
  if get_context().is_worker():
    assert target_role == DistRole.WORKER
  else:
    assert target_role in (DistRole.SERVER, DistRole.CLIENT)
  target_worker = _rpc_worker_names[target_role][role_rank]
  return rpc.rpc_async(to=target_worker, func=func, args=args, kwargs=kwargs)


@_require_initialized
def rpc_global_request(target_role: DistRole, role_rank: int,
                       func, args=None, kwargs=None):
  r""" Perform a rpc request synchronously to other rpc worker on
  arbitrary role group and return the results.
  """
  fut = rpc_global_request_async(target_role, role_rank, func, args, kwargs)
  return fut.wait()

@_require_initialized
def rpc_global_request_by_rank(global_rank: int, func, args=None, kwargs=None):
  r""" Perform a rpc request synchronously to other rpc worker by rank
  and return the results.
  """
  fut = rpc.rpc_async(global_rank, func, args, kwargs)
  return fut.wait()
