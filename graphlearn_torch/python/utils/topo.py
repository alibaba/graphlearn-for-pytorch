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

from typing import Optional, Tuple

import torch
import torch_sparse


def ptr2ind(ptr: torch.Tensor) -> torch.Tensor:
  r""" Convert an index pointer tensor to an indice tensor.
  """
  ind = torch.arange(ptr.numel() - 1, device=ptr.device)
  return ind.repeat_interleave(ptr[1:] - ptr[:-1])


def coo_to_csr(
  row: torch.Tensor,
  col: torch.Tensor,
  edge_id: Optional[torch.Tensor] = None,
  edge_weight: Optional[torch.Tensor] = None,
  node_sizes: Optional[Tuple[int, int]] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
  r""" Tranform edge index from COO to CSR.

  Args:
    row (torch.Tensor): The row indices.
    col (torch.Tensor): The column indices.
    edge_value (torch.Tensor, optional): The value indices corresponding to
      the input edge index.
    node_sizes (Tuple[int, int], optional): The number of nodes in row and col.
  """
  if node_sizes is None:
    node_sizes = (int(row.max()) + 1, int(col.max()) + 1)
    assert len(node_sizes) == 2
  assert row.numel() == col.numel()
  if edge_id is not None:
    assert edge_id.numel() == row.numel()
    edge_value = edge_id
  if edge_weight is not None:
    assert edge_weight.numel() == row.numel()
    edge_value = torch.stack((edge_id, edge_weight)).T
  adj_t = torch_sparse.SparseTensor(
    row=row, col=col, value=edge_value, sparse_sizes=node_sizes
  )
  edge_value = adj_t.storage.value()
  if len(edge_value.shape) == 2:
    edge_ids, edge_weights = edge_value[:,0], edge_value[:,1]
  elif edge_id is not None:
    edge_ids, edge_weights = edge_value, torch.empty(0)
  else:
    edge_ids, edge_weights = torch.empty(0), edge_value
  return adj_t.storage.rowptr(), adj_t.storage.col(), edge_ids, edge_weights


def coo_to_csc(
  row: torch.Tensor,
  col: torch.Tensor,
  edge_id: Optional[torch.Tensor] = None,
  edge_weight: Optional[torch.Tensor] = None,
  node_sizes: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
  r""" Tranform edge index from COO to CSC.

  Args:
    row (torch.Tensor): The row indices.
    col (torch.Tensor): The column indices.
    edge_value (torch.Tensor, optional): The value indices corresponding to
      the input edge index.
    node_sizes (Tuple[int, int], optional): The number of nodes in row and col.
  """
  if node_sizes is not None:
    node_sizes = (node_sizes[1], node_sizes[0])
  r_colptr, r_row, r_edge_id, r_edge_weight = coo_to_csr(
    col, row, edge_id, edge_weight, node_sizes
  )
  return r_row, r_colptr, r_edge_id, r_edge_weight
