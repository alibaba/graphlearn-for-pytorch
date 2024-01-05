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

from tqdm import tqdm

import graphlearn_torch as glt
from utils import get_gpt_response, link_prediction


def run(rank, glt_ds, raw_text, reason):
  neg_sampling = glt.sampler.NegativeSampling('binary')
  train_loader = glt.loader.LinkNeighborLoader(glt_ds,
                                              [12, 6],
                                              neg_sampling=neg_sampling,
                                              batch_size=2,
                                              drop_last=True,
                                              shuffle=True,
                                              device=torch.device(rank))
  print(f'Rank {rank} build graphlearn_torch NeighborLoader Done.')

  for batch in tqdm(train_loader):
    batch_titles = raw_text[batch.node]
    if batch.edge_index.shape[1] < 5:
      continue

    # print(batch)
    # print(batch.edge_label_index)
    message = link_prediction(batch, batch_titles, reason=reason)

    # print(message)
    response = get_gpt_response(
      message=message
    )

    print(f"response: {response}")


if __name__ == '__main__':
  world_size = torch.cuda.device_count()
  import pandas as pd
  root = '../data/arxiv_2023/raw/'
  titles = pd.read_csv(root + "titles.csv.gz").to_numpy()
  ids = torch.from_numpy(pd.read_csv(root + "ids.csv.gz").to_numpy())
  edge_index = torch.from_numpy(pd.read_csv(root + "edges.csv.gz").to_numpy())

  print('Build graphlearn_torch dataset...')
  start = time.time()
  glt_dataset = glt.data.Dataset()
  glt_dataset.init_graph(
    edge_index=edge_index.T,
    graph_mode='CPU',
    directed=True
  )
  glt_dataset.init_node_features(
    node_feature_data=ids,
    sort_func=glt.data.sort_by_in_degree,
    split_ratio=0
  )

  print(f'Build graphlearn_torch csr_topo and feature cost {time.time() - start} s.')

  run(0, glt_dataset, titles, reason=False)