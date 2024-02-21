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

try:
  import torch
  from typing import Dict
  from collections.abc import Sequence

  from .. import py_graphlearn_torch_vineyard as pywrap


except ImportError:
   pass

from ..partition import PartitionBook


def vineyard_to_csr(sock, fid, v_label_name, e_label_name, edge_dir, haseid=0):
  '''
    Wrap to_csr function to read graph from vineyard
    with return (indptr, indices, (Optional)edge_id)
  '''
  return pywrap.vineyard_to_csr(sock, fid, v_label_name, e_label_name, edge_dir, haseid)


def load_vertex_feature_from_vineyard(sock, fid, vcols, v_label_name):
  '''
    Wrap load_vertex_feature_from_vineyard function to read vertex feature
    from vineyard
    return vertex_feature(torch.Tensor)
  '''
  return pywrap.load_vertex_feature_from_vineyard(sock, fid, v_label_name, vcols)


def load_edge_feature_from_vineyard(sock, fid, ecols, e_label_name):
  '''
    Wrap load_edge_feature_from_vineyard function to read edge feature
    from vineyard
    return edge_feature(torch.Tensor)
  '''
  return pywrap.load_edge_feature_from_vineyard(sock, fid, e_label_name, ecols)


def get_fid_from_gid(gid):
  '''
    Wrap get_fid_from_gid function to get fid from gid
  '''
  return pywrap.get_fid_from_gid(gid)


def get_frag_vertex_offset(sock, fid, v_label_name):
  '''
    Wrap GetFragVertexOffset function to get vertex offset of a fragment.
  '''
  return pywrap.get_frag_vertex_offset(sock, fid, v_label_name)


def get_frag_vertex_num(sock, fid, v_label_name):
  '''
    Wrap GetFragVertexNum function to get vertex number of a fragment.
  '''
  return pywrap.get_frag_vertex_num(sock, fid, v_label_name)


class VineyardPartitionBook(PartitionBook):
  def __init__(self, sock, obj_id, v_label_name, fid2pid: Dict=None):
    self._sock = sock
    self._obj_id = obj_id
    self._v_label_name = v_label_name
    self._frag = None
    self._offset = get_frag_vertex_offset(sock, obj_id, v_label_name)
    # TODO: optimise this query process if too slow
    self._fid2pid = fid2pid

  def __getitem__(self, gids) -> torch.Tensor:
    fids = self.gid2fid(gids)
    if self._fid2pid is not None:
      pids = torch.tensor([self._fid2pid[fid] for fid in fids])
      return pids.to(torch.int32)
    return fids.to(torch.int32)

  @property
  def device(self):
    return torch.device('cpu')

  @property
  def offset(self):
    return self._offset

  def gid2fid(self, gids):
    '''
      Parse gid to get fid
    '''
    if self._frag is None:
      self._frag = pywrap.VineyardFragHandle(self._sock, self._obj_id)

    fids = self._frag.get_fid_from_gid(gids.tolist())

    return fids


class VineyardGid2Lid(Sequence):
  def __init__(self, sock, fid, v_label_name):
    self._offset = get_frag_vertex_offset(sock, fid, v_label_name)
    self._vnum = get_frag_vertex_num(sock, fid, v_label_name)

  def __getitem__(self, gids):
    return gids - self._offset
  
  def __len__(self):
    return self._vnum

def v6d_id_select(srcs, p_mask, node_pb: PartitionBook):
  '''
    Select the inner vertices in `srcs` that belong to a specific partition,
    and return their local offsets in the partition.
  '''
  gids = torch.masked_select(srcs, p_mask)
  offsets = gids - node_pb.offset
  return offsets

def v6d_id_filter(node_pb: VineyardPartitionBook, partition_idx):
  '''
    Select the inner vertices that belong to a specific partition
  '''
  frag = pywrap.VineyardFragHandle(node_pb._sock, node_pb._obj_id)
  inner_vertices = frag.get_inner_vertices(node_pb._v_label_name)
  return inner_vertices
