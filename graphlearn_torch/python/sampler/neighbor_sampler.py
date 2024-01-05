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

import math
from typing import Dict, Optional, Union, Literal

import torch
import threading

from .. import py_graphlearn_torch as pywrap
from ..data import Graph
from ..typing import NodeType, EdgeType, NumNeighbors, reverse_edge_type
from ..utils import (
  merge_dict, merge_hetero_sampler_output, format_hetero_sampler_output,
  id2idx, count_dict
)


from .base import (
  BaseSampler, EdgeIndex,
  NodeSamplerInput, EdgeSamplerInput,
  SamplerOutput, HeteroSamplerOutput, NeighborOutput,
)
from .negative_sampler import RandomNegativeSampler

class NeighborSampler(BaseSampler):
  r""" Neighbor Sampler.
  """
  def __init__(self,
               graph: Union[Graph, Dict[EdgeType, Graph]],
               num_neighbors: Optional[NumNeighbors] = None,
               device: torch.device=torch.device('cuda', 0),
               with_edge: bool=False,
               with_neg: bool=False,
               with_weight: bool=False,
               strategy: str = 'random',
               edge_dir: Literal['in', 'out'] = 'out',
               seed: int = None):
    self.graph = graph
    self.num_neighbors = num_neighbors
    self.device = device
    self.with_edge = with_edge
    self.with_neg = with_neg
    self.with_weight = with_weight
    self.strategy = strategy
    self.edge_dir = edge_dir
    self._subgraph_op = None
    self._sampler = None
    self._neg_sampler = None
    self._inducer = None
    self._sampler_lock = threading.Lock()
    self.is_sampler_initialized = False
    self.is_neg_sampler_initialized = False
    
    if seed is not None:
      pywrap.RandomSeedManager.getInstance().setSeed(seed)
    if isinstance(self.graph, Graph): #homo
      self._g_cls = 'homo'
      if self.graph.mode == 'CPU':
        self.device = torch.device('cpu')
    else: # hetero
      self._g_cls = 'hetero'
      self.edge_types = []
      self.node_types = set()
      for etype, graph in self.graph.items():
        self.edge_types.append(etype)
        self.node_types.add(etype[0])
        self.node_types.add(etype[2])
      if self.graph[self.edge_types[0]].mode == 'CPU':
        self.device = torch.device('cpu')
      self._set_num_neighbors_and_num_hops(self.num_neighbors)


  @property
  def subgraph_op(self):
    self.lazy_init_subgraph_op()
    return self._subgraph_op

  def lazy_init_sampler(self):
    if not self.is_sampler_initialized:
      with self._sampler_lock:
        if self._sampler is None:
          if self._g_cls == 'homo':
            if self.device.type == 'cuda':
              self._sampler = pywrap.CUDARandomSampler(self.graph.graph_handler)
            elif self.with_weight == False:
              self._sampler = pywrap.CPURandomSampler(self.graph.graph_handler)
            else:
              self._sampler = pywrap.CPUWeightedSampler(self.graph.graph_handler)
            self.is_sampler_initialized = True

          else: # hetero
            self._sampler = {}
            for etype, g in self.graph.items():
              if self.device != torch.device('cpu'):
                self._sampler[etype] = pywrap.CUDARandomSampler(g.graph_handler)
              elif self.with_weight == False:
                self._sampler[etype] = pywrap.CPURandomSampler(g.graph_handler)
              else:
                self._sampler[etype] = pywrap.CPUWeightedSampler(g.graph_handler)
            self.is_sampler_initialized = True


  def lazy_init_neg_sampler(self):
    if not self.is_neg_sampler_initialized and self.with_neg:
      with self._sampler_lock:
        if self._neg_sampler is None:
          if self._g_cls == 'homo':
            self._neg_sampler = RandomNegativeSampler(
              graph=self.graph,
              mode=self.device.type.upper(),
              edge_dir=self.edge_dir
            )
            self.is_neg_sampler_initialized = True
          else: # hetero
            self._neg_sampler = {}
            for etype, g in self.graph.items():
              self._neg_sampler[etype] = RandomNegativeSampler(
                graph=g,
                mode=self.device.type.upper(),
                edge_dir=self.edge_dir
              )
            self.is_neg_sampler_initialized = True

  def lazy_init_subgraph_op(self):
    if self._subgraph_op is None:
      with self._sampler_lock:
        if self._subgraph_op is None:
          if self.device.type == 'cuda':
            self._subgraph_op = pywrap.CUDASubGraphOp(self.graph.graph_handler)
          else:
            self._subgraph_op = pywrap.CPUSubGraphOp(self.graph.graph_handler)

  def sample_one_hop(
    self,
    input_seeds: torch.Tensor,
    req_num: int,
    etype: EdgeType = None
  ) -> NeighborOutput:
    self.lazy_init_sampler()
    sampler = self._sampler[etype] if etype is not None else self._sampler
    input_seeds = input_seeds.to(self.device)
    edge_ids = None

    if not self.with_edge:
      nbrs, nbrs_num = sampler.sample(input_seeds, req_num)
    else:
      nbrs, nbrs_num, edge_ids = sampler.sample_with_edge(input_seeds, req_num)

    if nbrs.numel() == 0:
      nbrs = torch.tensor([], dtype=torch.int64 ,device=self.device)
      nbrs_num = torch.zeros_like(input_seeds, dtype=torch.int64, device=self.device)
      edge_ids = torch.tensor([], device=self.device, dtype=torch.int64) \
        if self.with_edge else None
    return NeighborOutput(nbrs, nbrs_num, edge_ids)

  def sample_from_nodes(
    self,
    inputs: NodeSamplerInput,
    **kwargs
  ) -> Union[HeteroSamplerOutput, SamplerOutput]:
    inputs = NodeSamplerInput.cast(inputs)
    input_seeds = inputs.node.to(self.device)
    input_type = inputs.input_type

    if self._g_cls == 'hetero':
      assert input_type is not None
      output = self._hetero_sample_from_nodes({input_type: input_seeds})
    else:
      output = self._sample_from_nodes(input_seeds)
    return output


  def _sample_from_nodes(
    self,
    input_seeds: torch.Tensor
  ) -> SamplerOutput:
    r""" Sample on homogenous graphs and induce COO format subgraph.

    Note that messages in PyG are passed from src to dst. In 'out' direction,
    we sample src's out neighbors and induce [src_index, dst_index] subgraphs. 
    The direction of sampling is opposite to the direction of message passing. 
    To be consistent with the semantics of PyG, the final edge index is 
    transpose to [dst_index, src_index]. In 'in' direction, we don't need to 
    reverse it.
    """
    out_nodes, out_rows, out_cols, out_edges = [], [], [], []
    num_sampled_nodes, num_sampled_edges = [], []
    inducer = self.get_inducer(input_seeds.numel())
    srcs = inducer.init_node(input_seeds)
    batch = srcs
    num_sampled_nodes.append(input_seeds.numel())
    out_nodes.append(srcs)
    for req_num in self.num_neighbors:
      out_nbrs = self.sample_one_hop(srcs, req_num)
      if out_nbrs.nbr.numel() == 0:
        break
      nodes, rows, cols = inducer.induce_next(
        srcs, out_nbrs.nbr, out_nbrs.nbr_num)
      out_nodes.append(nodes)
      out_rows.append(rows)
      out_cols.append(cols)
      if out_nbrs.edge is not None:
        out_edges.append(out_nbrs.edge)
      num_sampled_nodes.append(nodes.size(0))
      num_sampled_edges.append(cols.size(0))
      srcs = nodes

    return SamplerOutput(
      node=torch.cat(out_nodes),
      row=torch.cat(out_cols) if len(out_cols) > 0 else torch.tensor(out_cols),
      col=torch.cat(out_rows) if len(out_rows) > 0 else torch.tensor(out_rows),
      edge=(torch.cat(out_edges) if out_edges else None),
      batch=batch,
      num_sampled_nodes=num_sampled_nodes,
      num_sampled_edges=num_sampled_edges,
      device=self.device
    )

  def _hetero_sample_from_nodes(
    self,
    input_seeds_dict: Dict[NodeType, torch.Tensor],
  ) -> HeteroSamplerOutput:
    r""" Sample on heterogenous graphs and induce COO format subgraph dict.

    Note that messages in PyG are passed from src to dst. In 'out' direction,
    we sample src's out neighbors and induce [src_index, dst_index] subgraphs. 
    The direction of sampling is opposite to the direction of message passing. 
    To be consistent with the semantics of PyG, the final edge index is transpose to
    [dst_index, src_index] and edge_type is reversed as well. For example,
    given the edge_type (u, u2i, i), we sample by meta-path u->i, but return
    edge_index_dict {(i, rev_u2i, u) : [i, u]}. In 'in' direction, we don't need to
    reverse it.
    """
    # sample neighbors hop by hop.
    max_input_batch_size = max([t.numel() for t in input_seeds_dict.values()])
    inducer = self.get_inducer(max_input_batch_size)
    src_dict = inducer.init_node(input_seeds_dict)
    batch = src_dict
    out_nodes, out_rows, out_cols, out_edges = {}, {}, {}, {}
    num_sampled_nodes, num_sampled_edges = {}, {}
    merge_dict(src_dict, out_nodes)
    count_dict(src_dict, num_sampled_nodes, 1)
    for i in range(self.num_hops):
      nbr_dict, edge_dict = {}, {}
      for etype in self.edge_types:
        req_num = self.num_neighbors[etype][i]
        # out sampling needs dst_type==seed_type, in sampling needs src_type==seed_type
        if self.edge_dir == 'in':
          src = src_dict.get(etype[-1], None)
          if src is not None and src.numel() > 0:
            output = self.sample_one_hop(src, req_num, etype)
            if output.nbr.numel() == 0:
              continue
            nbr_dict[reverse_edge_type(etype)] = [src, output.nbr, output.nbr_num]
            if output.edge is not None:
              edge_dict[reverse_edge_type(etype)] = output.edge
        elif self.edge_dir == 'out':
          src = src_dict.get(etype[0], None)
          if src is not None and src.numel() > 0:
            output = self.sample_one_hop(src, req_num, etype)
            if output.nbr.numel() == 0:
              continue
            nbr_dict[etype] = [src, output.nbr, output.nbr_num]
            if output.edge is not None:
              edge_dict[etype] = output.edge
      if len(nbr_dict) == 0:
        continue
      nodes_dict, rows_dict, cols_dict = inducer.induce_next(nbr_dict)
      merge_dict(nodes_dict, out_nodes)
      merge_dict(rows_dict, out_rows)
      merge_dict(cols_dict, out_cols)
      merge_dict(edge_dict, out_edges)
      count_dict(nodes_dict, num_sampled_nodes, i + 2)
      count_dict(cols_dict, num_sampled_edges, i + 1)
      src_dict = nodes_dict

    for etype, rows in out_rows.items():
      out_rows[etype] = torch.cat(rows)
      out_cols[etype] = torch.cat(out_cols[etype])
      if self.with_edge:
        out_edges[etype] = torch.cat(out_edges[etype])

    res_rows, res_cols, res_edges = {}, {}, {}
    for etype, rows in out_rows.items():
      rev_etype = reverse_edge_type(etype)
      res_rows[rev_etype] = out_cols[etype]
      res_cols[rev_etype] = rows
      if self.with_edge:
        res_edges[rev_etype] = out_edges[etype]

    return HeteroSamplerOutput(
      node={k : torch.cat(v) for k, v in out_nodes.items()},
      row=res_rows,
      col=res_cols,
      edge=(res_edges if len(res_edges) else None),
      batch=batch,
      num_sampled_nodes={k : torch.tensor(v, device=self.device)
        for k, v in num_sampled_nodes.items()},
      num_sampled_edges={
        reverse_edge_type(k) : torch.tensor(v, device=self.device)
        for k, v in num_sampled_edges.items()},
      edge_types=self.edge_types,
      device=self.device
    )

  def sample_from_edges(
    self,
    inputs: EdgeSamplerInput,
    **kwargs,
  ) -> Union[HeteroSamplerOutput, SamplerOutput]:
    r"""Performs sampling from an edge sampler input, leveraging a sampling
    function of the same signature as `node_sample`.

    Note that in out-edge sampling, we reverse the direction of src and dst
    for the output so that features of the sampled nodes during training can
    be aggregated from k-hop to (k-1)-hop nodes.
    """
    src = inputs.row.to(self.device)
    dst = inputs.col.to(self.device)
    edge_label = None if inputs.label is None else inputs.label.to(self.device)
    input_type = inputs.input_type
    neg_sampling = inputs.neg_sampling

    num_pos = src.numel()
    num_neg = 0
    # Negative Sampling
    self.lazy_init_neg_sampler()
    if neg_sampling is not None:
      # When we are doing negative sampling, we append negative information
      # of nodes/edges to `src`, `dst`.
      # Later on, we can easily reconstruct what belongs to positive and
      # negative examples by slicing via `num_pos`.
      num_neg = math.ceil(num_pos * neg_sampling.amount)
      if neg_sampling.is_binary():
        # In the "binary" case, we randomly sample negative pairs of nodes.
        if input_type is not None:
          neg_pair = self._neg_sampler[input_type].sample(num_neg)
        else:
          neg_pair = self._neg_sampler.sample(num_neg)
        src_neg, dst_neg = neg_pair[0], neg_pair[1]
        src = torch.cat([src, src_neg], dim=0)
        dst = torch.cat([dst, dst_neg], dim=0)
        if edge_label is None:
            edge_label = torch.ones(num_pos, device=self.device)
        size = (src_neg.size()[0], ) + edge_label.size()[1:]
        edge_neg_label = edge_label.new_zeros(size)
        edge_label = torch.cat([edge_label, edge_neg_label])
      elif neg_sampling.is_triplet():
        # TODO: make triplet negative sampling strict.
        # In the "triplet" case, we randomly sample negative destinations
        # in a "non-strict" manner.
        assert num_neg % num_pos == 0
        if input_type is not None:
          neg_pair = self._neg_sampler[input_type].sample(num_neg, padding=True)
        else:
          neg_pair = self._neg_sampler.sample(num_neg, padding=True)
        dst_neg = neg_pair[1]
        dst = torch.cat([dst, dst_neg], dim=0)
        assert edge_label is None
    # Neighbor Sampling
    if input_type is not None: # hetero
      if input_type[0] != input_type[-1]:  # Two distinct node types:
        src_seed, dst_seed = src, dst
        src, inverse_src = src.unique(return_inverse=True)
        dst, inverse_dst = dst.unique(return_inverse=True)
        seed_dict = {input_type[0]: src, input_type[-1]: dst}
      else:  # Only a single node type: Merge both source and destination.
        seed = torch.cat([src, dst], dim=0)
        seed, inverse_seed = seed.unique(return_inverse=True)
        seed_dict = {input_type[0]: seed}

      temp_out = []
      for it, node in seed_dict.items():
        seeds = NodeSamplerInput(node=node, input_type=it)
        temp_out.append(self.sample_from_nodes(seeds))
      if len(temp_out) == 2:
        out = merge_hetero_sampler_output(temp_out[0],
                                          temp_out[1],
                                          device=self.device,
                                          edge_dir=self.edge_dir)
      else:
        out = format_hetero_sampler_output(temp_out[0], edge_dir=self.edge_dir)
      # edge_label
      if neg_sampling is None or neg_sampling.is_binary():
        if input_type[0] != input_type[-1]:
          inverse_src = id2idx(out.node[input_type[0]])[src_seed]
          inverse_dst = id2idx(out.node[input_type[-1]])[dst_seed]
          edge_label_index = torch.stack([
              inverse_src,
              inverse_dst,
          ], dim=0)
        else:
          edge_label_index = inverse_seed.view(2, -1)

        out.metadata = {'edge_label_index': edge_label_index,
                        'edge_label': edge_label}
        out.input_type = input_type
      elif neg_sampling.is_triplet():
        if input_type[0] != input_type[-1]:
          inverse_src = id2idx(out.node[input_type[0]])[src_seed]
          inverse_dst = id2idx(out.node[input_type[-1]])[dst_seed]
          src_index = inverse_src
          dst_pos_index = inverse_dst[:num_pos]
          dst_neg_index = inverse_dst[num_pos:]
        else:
          src_index = inverse_seed[:num_pos]
          dst_pos_index = inverse_seed[num_pos:2 * num_pos]
          dst_neg_index = inverse_seed[2 * num_pos:]
        dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)

        out.metadata = {'src_index': src_index,
                        'dst_pos_index': dst_pos_index,
                        'dst_neg_index': dst_neg_index}
        out.input_type = input_type
    else: #homo
      seed = torch.cat([src, dst], dim=0)
      seed, inverse_seed = seed.unique(return_inverse=True)
      out = self.sample_from_nodes(seed)
      # edge_label
      if neg_sampling is None or neg_sampling.is_binary():
        edge_label_index = inverse_seed.view(2, -1)

        out.metadata = {'edge_label_index': edge_label_index,
                        'edge_label': edge_label}
      elif neg_sampling.is_triplet():
        src_index = inverse_seed[:num_pos]
        dst_pos_index = inverse_seed[num_pos:2 * num_pos]
        dst_neg_index = inverse_seed[2 * num_pos:]
        dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)
        out.metadata = {'src_index': src_index,
                        'dst_pos_index': dst_pos_index,
                        'dst_neg_index': dst_neg_index}
    return out

  def sample_pyg_v1(self, ids: torch.Tensor):
    r""" Sample multi-hop neighbors and organize results to PyG's `EdgeIndex`.

    Args:
      ids: input ids, 1D tensor.
      The sampled results that is the same as PyG's `NeighborSampler`(PyG v1)
    """
    ids = ids.to(self.device)
    adjs = []
    srcs = ids
    out_ids = ids
    batch_size = 0
    inducer = self.get_inducer(srcs.numel())
    for i, req_num in enumerate(self.num_neighbors):
      srcs = inducer.init_node(srcs)
      batch_size = srcs.numel() if i == 0 else batch_size
      out_nbrs = self.sample_one_hop(srcs, req_num)
      nodes, rows, cols = \
        inducer.induce_next(srcs, out_nbrs.nbr, out_nbrs.nbr_num)
      edge_index = torch.stack([cols, rows]) # we use csr instead of csc in PyG.
      out_ids = torch.cat([srcs, nodes])
      adj_size = torch.LongTensor([out_ids.size(0), srcs.size(0)])
      adjs.append(EdgeIndex(edge_index, out_nbrs.edge, adj_size))
      srcs = out_ids
    return batch_size, out_ids, adjs[::-1]

  def subgraph(
    self,
    inputs: NodeSamplerInput,
  ) -> SamplerOutput:
    self.lazy_init_subgraph_op()
    inputs = NodeSamplerInput.cast(inputs)
    input_seeds = inputs.node.to(self.device)
    if self.num_neighbors is not None:
      nodes = [input_seeds]
      for num in self.num_neighbors:
        nbr = self.sample_one_hop(nodes[-1], num).nbr
        nodes.append(torch.unique(nbr))
      nodes, mapping = torch.cat(nodes).unique(return_inverse=True)
    else:
      nodes, mapping = torch.unique(input_seeds, return_inverse=True)
    subgraph = self._subgraph_op.node_subgraph(nodes, self.with_edge)

    return SamplerOutput(
      node=subgraph.nodes,
      # The edge index should be reversed.
      row=subgraph.cols,
      col=subgraph.rows,
      edge=subgraph.eids if self.with_edge else None,
      device=self.device,
      metadata=mapping[:input_seeds.numel()])

  def sample_prob(
    self,
    inputs: NodeSamplerInput,
    node_cnt: Union[int, Dict[NodeType, int]]
  ) -> Union[torch.Tensor, Dict[NodeType, torch.Tensor]]:
    r""" Get the probability of each node being sampled.
    """
    self.lazy_init_sampler()
    inputs = NodeSamplerInput.cast(inputs)
    input_seeds = inputs.node.to(self.device)
    input_type = inputs.input_type
    if self._g_cls == 'hetero':
      assert input_type is not None
      output = self._hetero_sample_prob({input_type : input_seeds}, node_cnt)
    else:
      output = self._sample_prob(input_seeds, node_cnt)
    return output

  def _sample_prob(
    self,
    input_seeds: torch.Tensor,
    node_cnt: int
  ) -> torch.Tensor:
    last_prob = \
      torch.ones(node_cnt, device=self.device, dtype=torch.float32) * 0.01
    last_prob[input_seeds] = 1
    for req in self.num_neighbors:
      cur_prob = torch.zeros(node_cnt, device=self.device, dtype=torch.float32)
      self._sampler.cal_nbr_prob(
        req, last_prob, last_prob, self.graph.graph_handler, cur_prob
      )
      last_prob = cur_prob
    return last_prob

  def _hetero_sample_prob(
    self,
    input_seeds_dict: Dict[NodeType, torch.Tensor],
    node_dict: Dict[NodeType, int]
  ) -> Dict[NodeType, torch.Tensor]:
    probs = {}
    for ntype in node_dict.keys():
      probs[ntype] = []

    # calculate probs for each subgraph
    for i in range(self.num_hops):
      for etype in self.edge_types:
        req = self.num_neighbors[etype][i]
        # homogenous subgraph case
        if etype[0] == etype[2]:
          if len(probs[etype[0]]) == 0:
            last_prob = torch.ones(node_dict[etype[0]].size(0),
                                   device=self.device,
                                   dtype=torch.float32) * 0.005
            last_prob[input_seeds_dict[etype[0]]] = 1
          else:
            last_prob = self.aggregate_prob(probs[etype[0]],
                                            node_dict[etype[0]].size(0),
                                            device=self.device)

          cur_prob = torch.zeros(node_dict[etype[0]].size(0),
                                 device=self.device,
                                 dtype=torch.float32)
          self._sampler[etype].cal_nbr_prob(
            req, last_prob, last_prob,
            self._graph_dict[etype].graph_handler, cur_prob
          )
          last_prob = cur_prob
          probs[etype[0]].append(last_prob)

        # hetero bipartite graph case
        else:
          if len(probs[etype[0]]) == 0:
            last_prob = torch.ones(node_dict[etype[0]].size(0),
                                   device=self.device,
                                   dtype=torch.float32) * 0.005
            last_prob[input_seeds_dict[etype[0]]] = 1
          else:
            last_prob = self.aggregate_prob(probs[etype[0]],
                                            node_dict[etype[0]].size(0),
                                            device=self.device)

          etypes = [nbr_etype
                    for nbr_etype in self.edge_types
                    if nbr_etype[0] == etype[2]]

          temp_probs = []
          # prepare nbr_prob
          if len(probs[etype[2]]) == 0:
            nbr_prob = torch.ones(node_dict[etype[2]].size(0),
                                  device=self.device,
                                  dtype=torch.float32) * 0.005
            if etype[2] in input_seeds_dict:
              nbr_prob[input_seeds_dict[etype[2]]] = 1
          else:
            nbr_prob = self.aggregate_prob(probs[etype[2]],
                                           node_dict[etype[2]].size(0),
                                           device=self.device)

          for nbr_etype in etypes:
            cur_prob = torch.zeros(node_dict[etype[0]].size(0),
                                   device=self.device,
                                   dtype=torch.float32)
            self._sampler[etype].cal_nbr_prob(
              req, last_prob, nbr_prob,
              self._graph_dict[nbr_etype].graph_handler, cur_prob
            )
            last_prob = cur_prob
            temp_probs.append(last_prob)

          # aggregate prob for the bipartite graph
          # with #{subgraphs where the neighbours are}
          sub_temp_prob = self.aggregate_prob(temp_probs,
                                              node_dict[etype[0]].size(0),
                                              device=self.device)

          probs[etype[0]].append(sub_temp_prob)

      # aggregate probs from each subgraph
      # with #{subgraphs}
      for ntype, prob in probs.items():
        res = self.aggregate_prob(
          prob, node_dict[ntype].size(0), device=self.device)
        if i == self.num_hops - 1:
          probs[ntype] = res
        else:
          probs[ntype] = [res]

    return probs

  def get_inducer(self, input_batch_size: int):
    if self._inducer is None:
      self._inducer = self.create_inducer(input_batch_size)
    return self._inducer

  def create_inducer(self, input_batch_size: int):
    max_num_nodes = self._max_sampled_nodes(input_batch_size)
    if self.device.type == 'cuda':
      if self._g_cls == 'homo':
        inducer = pywrap.CUDAInducer(max_num_nodes)
      else:
        inducer = pywrap.CUDAHeteroInducer(max_num_nodes)
    else:
      if self._g_cls == 'homo':
        inducer = pywrap.CPUInducer(max_num_nodes)
      else:
        inducer = pywrap.CPUHeteroInducer(max_num_nodes)
    return inducer

  def _set_num_neighbors_and_num_hops(self, num_neighbors):
    if isinstance(num_neighbors, (list, tuple)):
      num_neighbors = {key: num_neighbors for key in self.edge_types}
    assert isinstance(num_neighbors, dict)
    self.num_neighbors = num_neighbors
    # Add at least one element to the list to ensure `max` is well-defined
    self.num_hops = max([0] + [len(v) for v in num_neighbors.values()])
    for key, value in self.num_neighbors.items():
      if len(value) != self.num_hops:
        raise ValueError(f"Expected the edge type {key} to have "
                         f"{self.num_hops} entries (got {len(value)})")

  def _max_sampled_nodes(
    self,
    input_batch_size: int,
  ) -> Union[int, Dict[str, int]]:
    if self._g_cls == 'homo':
      res = [input_batch_size]
      for num in self.num_neighbors:
        res.append(res[-1] * num)
      return sum(res)

    res = {k : [] for k in self.node_types}
    for etype, num_list in self.num_neighbors.items():
      tmp_res = [input_batch_size]
      for num in num_list:
        tmp_res.append(tmp_res[-1] * num)
      res[etype[0]].extend(tmp_res)
      res[etype[2]].extend(tmp_res)
    return {k : sum(v) for k, v in res.items()}

  def _aggregate_prob(self, probs, node_num, device):
    """
      Aggregate probs from each subgraph
      p = 1 - ((1-p_0)(1-p_1)...(1-p_k))**(1/k)
      where k := #{subgraphs}
    """

    res = torch.ones(node_num, device=device, dtype=torch.float32)
    for temp_prob in probs:
      # to avoid the case that p_i=1 causes p=1 s.t the whole importance won't
      # be decided by one term.
      res *= (1 + .002 - temp_prob)
    res = 1 - res ** (1/len(probs))
    return res.clamp(min=0.0)
