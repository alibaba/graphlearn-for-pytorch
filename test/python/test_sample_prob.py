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
import os.path as osp
import unittest

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG

from graphlearn_torch.data import Topology, Graph
from graphlearn_torch.sampler import NeighborSampler, NodeSamplerInput


class SampleProbTestCase(unittest.TestCase):
  @unittest.skip("Download too long")
  def setUp(self):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../../data/')
    os.system('mkdir '+path+ ' && wget -P '+path+'mag/raw \
      https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/mag.zip \
      && wget -P '+path+ 'mag/raw \
      https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/mag_metapath2vec_emb.zip'
    )

    transform = T.ToUndirected(merge=True)
    dataset = OGB_MAG(path, preprocess='metapath2vec', transform=transform)
    data = dataset[0]

    # init graphlearn_torch Dataset.
    edge_dict, self.node_dict, csr_dict, self.graph_dict = {}, {}, {}, {}
    self.req_nums, self.ids = {}, {}
    for etype in data.edge_types:
      edge_dict[etype] = data[etype]['edge_index']

    for ntype in data.node_types:
      self.node_dict[ntype] = torch.tensor(list(range(len(data[ntype].x))))
    self.input_type = data.node_types[0]
    self.ids = torch.randperm(self.node_dict[self.input_type].size(0))[:5]

    for etype, eidx in edge_dict.items():
      csr_dict[etype] = Topology(edge_index=eidx)
      self.graph_dict[etype] = Graph(csr_topo=csr_dict[etype])
      self.req_nums[etype] = [5, 5]
      # print(f"{etype}: #row={self.graph_dict[etype].row_count} \
      #   #edge={self.graph_dict[etype].edge_count} \
      #   #col={self.graph_dict[etype].col_count}")

  @unittest.skip("Download too long")
  def test_sample_prob(self):
    sampler = NeighborSampler(self.graph_dict, self.req_nums)
    print("loading done!")

    inputs = NodeSamplerInput(
      node=self.ids,
      input_type=self.input_type
    )
    probs = sampler.sample_prob(inputs, self.node_dict)

    print(probs)
    assert(probs['paper'].size(0) == self.node_dict['paper'].size(0))
    assert(probs['author'].size(0) == self.node_dict['author'].size(0))
    assert(probs['field_of_study'].size(0) ==
           self.node_dict['field_of_study'].size(0))
    assert(probs['institution'].size(0) == self.node_dict['institution'].size(0))


if __name__ == "__main__":
  unittest.main()