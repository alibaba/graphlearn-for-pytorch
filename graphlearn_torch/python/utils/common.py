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
import socket
from typing import Any, Dict
from ..typing import reverse_edge_type

import torch

def ensure_dir(dir_path: str):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)


def merge_dict(in_dict: Dict[Any, Any], out_dict: Dict[Any, Any]):
  for k, v in in_dict.items():
    vals = out_dict.get(k, [])
    vals.append(v)
    out_dict[k] = vals


def get_free_port(host: str = 'localhost') -> int:
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.bind((host, 0))
  port = s.getsockname()[1]
  s.close()
  return port


def id2idx_v2(gid, book):
  if not isinstance(book, torch.Tensor):
    book = torch.tensor(book, dtype=torch.int64)
  max_id = torch.max(book).item()
  id2idx = torch.zeros(max_id + 1, dtype=torch.int64, device=book.device)
  id2idx[book] = torch.arange(book.size(0), dtype=torch.int64, device=book.device)

  return id2idx[gid]


def index_select(data, index):
  if data is None:
    return None
  if isinstance(data, dict):
    new_data = {}
    for k, v in data.items():
      new_data[k] = index_select(v, index)
    return new_data
  if isinstance(data, list):
    new_data = []
    for v in data:
      new_data.append(index_select(v, index))
    return new_data
  if isinstance(data, tuple):
    return tuple(index_select(list(data), index))
  if isinstance(index, tuple):
    start, end = index
    return data[start:end]
  return data[index]


def merge_hetero_sampler_output(in_sample: Any, out_sample: Any, device):
  def subid2gid(sample):
    for k, v in sample.row.items():
      sample.row[k] = sample.node[k[0]][v]
    for k, v in sample.col.items():
      sample.col[k] = sample.node[k[-1]][v]

  def merge_tensor_dict(in_dict, out_dict, unique=False):
    for k, v in in_dict.items():
      vals = out_dict.get(k, torch.tensor([], device=device))
      out_dict[k] = torch.cat((vals, v)).unique() if unique \
        else torch.cat((vals, v))

  subid2gid(in_sample)
  subid2gid(out_sample)
  merge_tensor_dict(in_sample.node, out_sample.node, unique=True)
  merge_tensor_dict(in_sample.row, out_sample.row)
  merge_tensor_dict(in_sample.col, out_sample.col)

  for k, v in out_sample.row.items():
    out_sample.row[k] = id2idx_v2(v, out_sample.node[k[0]])
  for k, v in out_sample.col.items():
    out_sample.col[k] = id2idx_v2(v, out_sample.node[k[-1]])

  # if in_sample.batch is not None and out_sample.batch is not None:
  #   merge_tensor_dict(in_sample.batch, out_sample.batch)
  if in_sample.edge is not None and out_sample.edge is not None:
    merge_tensor_dict(in_sample.edge, out_sample.edge, unique=False)
  if out_sample.edge_types is not None and in_sample.edge_types is not None:
    out_sample.edge_types = list(set(out_sample.edge_types) | set(in_sample.edge_types))
    out_sample.edge_types = [
      reverse_edge_type(etype) if etype[0] != etype[-1] else etype
      for etype in out_sample.edge_types
    ]

  return out_sample


def format_hetero_sampler_output(in_sample: Any):
  for k in in_sample.node.keys():
    in_sample.node[k] = in_sample.node[k].unique()
  if in_sample.edge_types is not None:
    in_sample.edge_types = [
      reverse_edge_type(etype) if etype[0] != etype[-1] else etype
      for etype in in_sample.edge_types
    ]
  return in_sample