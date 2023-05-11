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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eithPer express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import datetime
from multiprocessing.reduction import ForkingPickler
import numpy as np
import torch
import time

try:
  import common_io
except ImportError:
  pass

from .dataset import Dataset


class TableDataset(Dataset):
  def load(self,
           edge_tables=None,
           node_tables=None,
           graph_mode='ZERO_COPY',
           sort_func=None,
           split_ratio=0.0,
           device_group_list=None,
           directed=True,
           reader_threads=10,
           reader_capacity=10240,
           reader_batch_size=1024,
           label=None,
           device=None,
           **kwargs):
    """ Creates `Dataset` from ODPS tables.

    Args:
      edge_tables: A dict({edge_type : odps_table}) denoting each
        bipartite graph input table of heterogeneous graph, where edge_type is
        a tuple of (src_type, edge_type, dst_type).
      node_tables: A dict({node_type(str) : odps_table}) denoting each
        input node table.
      graph_mode: mode in graphlearn_torch's `Graph`, 'CPU', 'ZERO_COPY'
        or 'CUDA'.
      sort_func: function for feature reordering, return feature data(2D tenosr)
        and a map(1D tensor) from id to index.
      split_ratio: The proportion of data allocated to the GPU, between 0 and 1.
      device_group_list: A list of `DeviceGroup`. Each DeviceGroup must have the
        same size. A group of GPUs with peer-to-peer access to each other should
        be set in the same device group for high feature collection performance.
      directed: A Boolean value indicating whether the graph topology is
        directed.
      reader_threads: The number of threads of table reader.
      reader_capacity: The capacity of table reader.
      reader_batch_size: The number of records read at once.
      label: A CPU torch.Tensor(homo) or a Dict[NodeType, torch.Tensor](hetero)
        with the label data for graph nodes.
      device: The target cuda device rank to perform graph operations and
        feature lookups.
    """
    assert isinstance(edge_tables, dict)
    assert isinstance(node_tables, dict)
    edge_index, feature = {}, {}
    edge_hetero = (len(edge_tables) > 1)
    node_hetero = (len(node_tables) > 1)

    print("Start Loading edge and node tables...")
    step = 0
    start_time = time.time()
    for e_type, table in edge_tables.items():
      edge_list = []
      reader = common_io.table.TableReader(table,
                                           num_threads=reader_threads,
                                           capacity=reader_capacity)
      while True:
        try:
          data = reader.read(reader_batch_size, allow_smaller_final_batch=True)
          edge_list.extend(data)
          step += 1
        except common_io.exception.OutOfRangeException:
          reader.close()
          break
        if step % 1000 == 0:
          print(f"{datetime.datetime.now()}: load "
                f"{step * reader_batch_size} edges.")
      rows = [e[0] for e in edge_list]
      cols = [e[1] for e in edge_list]
      edge_array = np.stack([np.array(rows, dtype=np.int64),
                             np.array(cols, dtype=np.int64)])
      if edge_hetero:
        edge_index[e_type] = edge_array
      else:
        edge_index = edge_array
      del rows
      del cols
      del edge_list

    step = 0
    for n_type, table in node_tables.items():
      feature_list = []
      reader = common_io.table.TableReader(table,
                                           num_threads=reader_threads,
                                           capacity=reader_capacity)
      while True:
        try:
          data = reader.read(reader_batch_size, allow_smaller_final_batch=True)
          feature_list.extend(data)
          step += 1
        except common_io.exception.OutOfRangeException:
          reader.close()
          break
        if step % 1000 == 0:
          print(f"{datetime.datetime.now()}: load "
                f"{step * reader_batch_size} nodes.")
      ids = torch.tensor([feat[0] for feat in feature_list], dtype=torch.long)
      _, original_index = torch.sort(ids)
      if isinstance(feature_list[0][1], bytes):
        float_feat= [
          list(map(float, feat[1].decode().split(':')))
          for feat in feature_list
        ]
      else:
        float_feat= [
          list(map(float, feat[1].split(':')))
          for feat in feature_list
        ]
      if node_hetero:
        feature[n_type] = torch.tensor(float_feat)[original_index]
      else:
        feature = torch.tensor(float_feat)[original_index]
      del ids
      del original_index
      del float_feat
      del feature_list
    load_time = (time.time() - start_time) / 60
    print(f'Loading table completed in {load_time:.2f} minutes.')
    self.init_graph(edge_index, None, 'COO', graph_mode, directed, device)
    self.init_node_features(feature, None, sort_func, split_ratio,
                            device_group_list, device)
    self.init_node_labels(label)


## Pickling Registration

def rebuild_table_dataset(ipc_handle):
  ds = TableDataset.from_ipc_handle(ipc_handle)
  return ds

def reduce_table_dataset(dataset: TableDataset):
  ipc_handle = dataset.share_ipc()
  return (rebuild_table_dataset, (ipc_handle, ))

ForkingPickler.register(TableDataset, reduce_table_dataset)