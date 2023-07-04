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

import torch

from .. import py_graphlearn_torch as pywrap


class RandomNegativeSampler(object):
  r""" Random negative Sampler.

  Args:
    graph: A ``graphlearn_torch.data.Graph`` object.
    mode: Execution mode of sampling, 'CUDA' means sampling on
      GPU, 'CPU' means sampling on CPU.
    edge_dir: The direction of edges to be sampled, determines 
      the order of rows and columns returned.
  """
  def __init__(self, graph, mode='CUDA', edge_dir='out'):
    self._mode = mode
    self.edge_dir = edge_dir
    if mode == 'CUDA':
      self._sampler = pywrap.CUDARandomNegativeSampler(graph.graph_handler)
    else:
      self._sampler = pywrap.CPURandomNegativeSampler(graph.graph_handler)

  def sample(self, req_num, trials_num=5, padding=False):
    r""" Negative sampling.

    Args:
      req_num: The number of request(max) negative samples.
      trials_num: The number of trials for negative sampling.
      padding: Whether to patch the negative sampling results to req_num.
        If True, after trying trials_num times, if the number of true negative
        samples is still less than req_num, just random sample edges(non-strict
        negative) as negative samples.

    Returns:
      negative edge_index(non-strict when padding is True).
    """
    if self.edge_dir == 'out':
      rows, cols = self._sampler.sample(req_num, trials_num, padding)
    elif self.edge_dir == 'in':
      cols, rows = self._sampler.sample(req_num, trials_num, padding)
    return torch.stack([rows, cols], dim=0)
