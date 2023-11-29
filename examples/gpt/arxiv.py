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

import time
import torch

import numpy as np
import os.path as osp

from numpy import genfromtxt

from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm

import graphlearn_torch as glt
from utils import get_gpt_response

def run(rank, glt_ds, train_idx):
  train_loader = glt.loader.NeighborLoader(glt_ds,
                                           [15, 10, 5],
                                           train_idx,
                                           batch_size=1,
                                           drop_last=True,
                                           shuffle=True,
                                           device=torch.device(rank))
  print(f'Rank {rank} build graphlearn_torch NeighborLoader Done.')

  for batch in tqdm(train_loader):
    # print(batch)
    if batch.edge_index.shape[1] < 15:
      continue
    message = "This is a directed subgraph of arxiv citation network with " + str(batch.x.shape[0]) + " nodes numbered from 0 to " + str(batch.x.shape[0]-1) + ".\n"
    message += "The subgraph has " + str(batch.edge_index.shape[1]) + " edges.\n"
    for i in range(1, batch.x.shape[0]):
      feature_str = ','.join(f'{it:.3f}' for it in batch.x[i].tolist())
      message += "The feature of node " + str(i) + " is [" + feature_str + "] "
      message += "and the node label is " + str(batch.y[i].item()) + ".\n"
    # message += "The edges of the subgraph are " + str(batch.edge_index.T.tolist()) + ' where the first number indicates source node and the second destination node.\n'
    message += "Question: predict the label for node 0, whose feature is [" + ','.join(f'{it:.3f}' for it in batch.x[0].tolist()) + "]. Give the label only and don't show any reasoning process.\n\n"
    # print(message)
    response = get_gpt_response(
      api_key='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',
      message=message
    )

    print(f"response: {response} label: {batch.y[0]}")


if __name__ == '__main__':
  world_size = torch.cuda.device_count()
  start = time.time()
  root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'arxiv')
  dataset = PygNodePropPredDataset('ogbn-arxiv', root)
  split_idx = dataset.get_idx_split()
  data = dataset[0]
  train_idx = split_idx['train']
  print(f'Load data cost {time.time()-start} s.')

  start = time.time()
  print('Build graphlearn_torch dataset...')
  glt_dataset = glt.data.Dataset()
  glt_dataset.init_graph(
    edge_index=data.edge_index,
    graph_mode='CPU',
    directed=False
  )
  glt_dataset.init_node_features(
    node_feature_data=data.x,
    sort_func=glt.data.sort_by_in_degree,
    split_ratio=0
  )
  glt_dataset.init_node_labels(node_label_data=data.y)
  print(f'Build graphlearn_torch csr_topo and feature cost {time.time() - start} s.')

  run(0, glt_dataset, train_idx)