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
from typing import Any, Dict, Callable, Optional, Literal
from ..typing import reverse_edge_type
from .tensor import id2idx

import numpy
import random
import torch
import pickle

def ensure_dir(dir_path: str):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  
def merge_dict(in_dict: Dict[Any, Any], out_dict: Dict[Any, Any]):
  for k, v in in_dict.items():
    vals = out_dict.get(k, [])
    vals.append(v)
    out_dict[k] = vals

def count_dict(in_dict: Dict[Any, Any], out_dict: Dict[Any, Any], target_len):
  for k, v in in_dict.items():
    vals = out_dict.get(k, [])
    vals += [0] * (target_len - len(vals) - 1)
    vals.append(len(v))
    out_dict[k] = vals

def get_free_port(host: str = 'localhost') -> int:
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((host, 0))
  port = s.getsockname()[1]
  s.close()
  return port


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


def merge_hetero_sampler_output(
    in_sample: Any, out_sample: Any, device,
    edge_dir: Literal['in', 'out']='out'):
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
    out_sample.row[k] = id2idx(out_sample.node[k[0]])[v.to(torch.int64)]
  for k, v in out_sample.col.items():
    out_sample.col[k] = id2idx(out_sample.node[k[-1]])[v.to(torch.int64)]

  # if in_sample.batch is not None and out_sample.batch is not None:
  #   merge_tensor_dict(in_sample.batch, out_sample.batch)
  if in_sample.edge is not None and out_sample.edge is not None:
    merge_tensor_dict(in_sample.edge, out_sample.edge, unique=False)
  if out_sample.edge_types is not None and in_sample.edge_types is not None:
    out_sample.edge_types = list(set(out_sample.edge_types) | set(in_sample.edge_types))
    if edge_dir == 'out':
      out_sample.edge_types = [
        reverse_edge_type(etype) if etype[0] != etype[-1] else etype
        for etype in out_sample.edge_types
      ]

  return out_sample


def format_hetero_sampler_output(in_sample: Any, edge_dir=Literal['in', 'out']):
  for k in in_sample.node.keys():
    in_sample.node[k] = in_sample.node[k].unique()
  if in_sample.edge_types is not None:
    if edge_dir == 'out':
      in_sample.edge_types = [
        reverse_edge_type(etype) if etype[0] != etype[-1] else etype
        for etype in in_sample.edge_types
      ]
  return in_sample

# Append a tensor to a file using pickle
def append_tensor_to_file(filename, tensor):
    # Try to open file in append binary mode
    try:
        with open(filename, 'ab') as f:
            pickle.dump(tensor, f)
    except Exception as e:
        print('Error:', e)


# Load a file containing tensors and concatenate them into a single tensor
def load_and_concatenate_tensors(filename, device):
    # Load file and read tensors
    with open(filename, 'rb') as f:
        tensor_list = []
        while True:
            try:
                tensor = pickle.load(f)
                tensor_list.append(tensor)
            except EOFError:
                break
    # Pre-allocate memory for combined tensor
    combined_tensor = torch.empty((sum(t.shape[0] for t in tensor_list), 
      *tensor_list[0].shape[1:]), dtype=tensor_list[0].dtype, device=device)
    # Concatenate tensors in list into combined tensor
    start_idx = 0
    for tensor in tensor_list:
        end_idx = start_idx + tensor.shape[0]
        combined_tensor[start_idx:end_idx] = tensor.to(device)
        start_idx = end_idx
    return combined_tensor

## Default function to select ids in `srcs` that belong to a specific partition
def default_id_select(srcs, p_mask, node_pb=None):
   return torch.masked_select(srcs, p_mask)

## Default function to filter src ids in a specific partition from the partition book
def default_id_filter(node_pb, partition_idx):
  return torch.where(node_pb == partition_idx)[0]

def save_ckpt(
  ckpt_seq: int,
  ckpt_dir: str,
  model: torch.nn.Module,
  optimizer: Optional[torch.optim.Optimizer] = None,
  epoch: float = 0,
):
  """
  Saves a checkpoint of the model's state.

  Parameters:
  ckpt_seq (int): The sequence number of the checkpoint.
  ckpt_dir (str): The directory where the checkpoint will be saved.
  model (torch.nn.Module): The model to be saved.
  optimizer (Optional[torch.optim.Optimizer]): The optimizer, if any.
  epoch (float): The current epoch. Default is 0.
  """
  if not os.path.isdir(ckpt_dir):
      os.makedirs(ckpt_dir)
  ckpt_path = os.path.join(ckpt_dir, f"model_seq_{ckpt_seq}.ckpt")

  ckpt = {
      'seq': ckpt_seq,
      'epoch': epoch,
      'model_state_dict': model.state_dict()
  }
  if optimizer:
    ckpt['optimizer_state_dict'] = optimizer.state_dict()
  
  torch.save(ckpt, ckpt_path)

def load_ckpt(
  ckpt_seq: int,
  ckpt_dir: str,
  model: torch.nn.Module,
  optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
  """
  Loads a checkpoint of the model's state, returns the epoch of the checkpoint.

  Parameters:
  ckpt_seq (int): The sequence number of the checkpoint.
  ckpt_dir (str): The directory where the checkpoint will be saved.
  model (torch.nn.Module): The model to be saved.
  optimizer (Optional[torch.optim.Optimizer]): The optimizer, if any.
  """

  ckpt_path = os.path.join(ckpt_dir, f"model_seq_{ckpt_seq}.ckpt")
  try:
    ckpt = torch.load(ckpt_path)
  except FileNotFoundError:
    return -1

  model.load_state_dict(ckpt['model_state_dict'])
  epoch = ckpt.get('epoch')
  if optimizer:
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
  return epoch
