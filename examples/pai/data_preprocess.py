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

import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset


import numpy as np
from ogb.nodeproppred import NodePropPredDataset

# load data
root = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                '..', 'data', 'products')
dataset = NodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train']
graph, label = dataset[0] # label with shape(2449029, 1)
num_nodes = graph['num_nodes'] # 2449029
node_feat = graph['node_feat'] # shape(2449029, 100)
edge_index = graph['edge_index'] # shape(2, 123718280)

# dump to disk
train_table = osp.join(root, 'ogbn_products_train')
node_table = osp.join(root, 'ogbn_products_node')
edge_table = osp.join(root, 'ogbn_products_edge')

with open(train_table, 'w') as f:
  for i in train_idx:
    f.write(str(i) + '\t' + str(label[i][0]) + '\n')

with open(node_table, 'w') as f:
  for i in range(num_nodes):
    f.write(str(i) + '\t' + str(':'.join(map(str, node_feat[i]))) + '\n')

with open(edge_table, 'w') as f:
  for i in range(edge_index.shape[1]):
    f.write(str(edge_index[0][i]) + '\t' + str(edge_index[1][i]) + '\n')