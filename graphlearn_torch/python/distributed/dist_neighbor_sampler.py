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
import queue
from dataclasses import dataclass
from typing import List, Literal, Optional, Union, Tuple, Dict

import torch

from .. import py_graphlearn_torch as pywrap
from ..channel import ChannelBase, SampleMessage
from ..data import Feature
from ..sampler import (
  NodeSamplerInput, EdgeSamplerInput,
  NeighborOutput, SamplerOutput, HeteroSamplerOutput,
  NeighborSampler
)
from ..typing import EdgeType, as_str, NumNeighbors, reverse_edge_type, TensorDataType
from ..utils import (
    get_available_device, ensure_device, merge_dict, id2idx,
    merge_hetero_sampler_output, format_hetero_sampler_output, count_dict
)

from .dist_dataset import DistDataset
from .dist_feature import DistFeature
from .dist_graph import DistGraph
from .event_loop import ConcurrentEventLoop, wrap_torch_future
from .rpc import (
  RpcCalleeBase, rpc_register, rpc_request_async,
  RpcDataPartitionRouter, rpc_sync_data_partitions
)


@dataclass
class PartialNeighborOutput:
  r""" The sampled neighbor output of a subset of the original ids.

  * index: the index of the subset vertex ids.
  * output: the sampled neighbor output.
  """
  index: torch.Tensor
  output: NeighborOutput


class RpcSamplingCallee(RpcCalleeBase):
  r""" A wrapper for rpc callee that will perform rpc sampling from
  remote processes.
  """
  def __init__(self, sampler: NeighborSampler, device: torch.device):
    super().__init__()
    self.sampler = sampler
    self.device = device

  def call(self, *args, **kwargs):
    ensure_device(self.device)
    output = self.sampler.sample_one_hop(*args, **kwargs)
    if output is None:
      nbrs = torch.tensor([], dtype=torch.int64, device=torch.device('cpu'))
      nbrs_num = torch.zeros_like(args[0], dtype=torch.int64,
                                  device=torch.device('cpu'))
      edge_ids = torch.tensor([], device=torch.device('cpu'), dtype=torch.int64) \
        if self.with_edge else None
      return NeighborOutput(nbrs, nbrs_num, edge_ids)
    return output.to(torch.device('cpu'))

class RpcSubGraphCallee(RpcCalleeBase):
  r""" A wrapper for rpc callee that will perform rpc sampling from
  remote processes.
  """
  def __init__(self, sampler: NeighborSampler, device: torch.device):
    super().__init__()
    self.sampler = sampler
    self.device = device

  def call(self, *args, **kwargs):
    ensure_device(self.device)
    with_edge = kwargs['with_edge']
    output = self.sampler.subgraph_op.node_subgraph(args[0].to(self.device),
                                                    with_edge)
    eids = output.eids.to('cpu') if with_edge else None
    return output.nodes.to('cpu'), output.rows.to('cpu'), output.cols.to('cpu'), eids

class DistNeighborSampler(ConcurrentEventLoop):
  r""" Asynchronized and distributed neighbor sampler.

  Args:
    data (DistDataset): The graph and feature data with partition info.
    num_neighbors (NumNeighbors): The number of sampling neighbors on each hop.
    with_edge (bool): Whether to sample with edge ids. (default: False).
    with_neg (bool): Whether to do negative sampling. (default: False)
    edge_dir (str:["in", "out"]): The edge direction for sampling.
      Can be either :str:`"out"` or :str:`"in"`.
      (default: :str:`"out"`)
    collect_features (bool): Whether collect features for sampled results.
      (default: False).
    channel (ChannelBase, optional): The message channel to send sampled
      results. If set to `None`, the sampled results will be returned
      directly with `sample_from_nodes`. (default: ``None``).
    use_all2all (bool): Whether use all2all API to collect cross nodes'
      feature. (deafult: False)
    concurrency (int): The max number of concurrent seed batches processed by
      the current sampler. (default: ``1``).
    device: The device to use for sampling. If set to ``None``, the current
      cuda device (got by ``torch.cuda.current_device``) will be used if
      available, otherwise, the cpu device will be used. (default: ``None``).
  """
  def __init__(self,
               data: DistDataset,
               num_neighbors: Optional[NumNeighbors] = None,
               with_edge: bool = False,
               with_neg: bool = False,
               with_weight: bool = False,
               edge_dir: Literal['in', 'out'] = 'out',
               collect_features: bool = False,
               channel: Optional[ChannelBase] = None,
               use_all2all: bool = False,
               concurrency: int = 1,
               device: Optional[torch.device] = None,
               seed:int = None):
    self.data = data
    self.use_all2all = use_all2all
    self.num_neighbors = num_neighbors
    self.max_input_size = 0
    self.with_edge = with_edge
    self.with_neg = with_neg
    self.with_weight = with_weight
    self.edge_dir = edge_dir
    self.collect_features = collect_features
    self.channel = channel
    self.concurrency = concurrency
    self.device = get_available_device(device)
    self.seed = seed

    if isinstance(data, DistDataset):
      partition2workers = rpc_sync_data_partitions(
        num_data_partitions=self.data.num_partitions,
        current_partition_idx=self.data.partition_idx
      )
      self.rpc_router = RpcDataPartitionRouter(partition2workers)

      self.dist_graph = DistGraph(
        data.num_partitions, data.partition_idx,
        data.graph, data.node_pb, data.edge_pb
      )

      self.dist_node_feature = None
      self.dist_edge_feature = None
      if self.collect_features:
        if data.node_features is not None:
          self.dist_node_feature = DistFeature(
            data.num_partitions, data.partition_idx,
            data.node_features, data.node_feat_pb,
            local_only=False, rpc_router=self.rpc_router, device=self.device
          )
        if self.with_edge and data.edge_features is not None:
          self.dist_edge_feature = DistFeature(
            data.num_partitions, data.partition_idx,
            data.edge_features, data.edge_feat_pb,
            local_only=False, rpc_router=self.rpc_router, device=self.device
          )
      # dist_node_labels should is initialized as a DistFeature object in the v6d case
      self.dist_node_labels = self.data.node_labels
      if self.dist_graph.data_cls == 'homo':
        if self.dist_node_labels is not None and \
            not isinstance(self.dist_node_labels, torch.Tensor):
          self.dist_node_labels = DistFeature(
            self.data.num_partitions, self.data.partition_idx,
            self.dist_node_labels, self.data.node_feat_pb,
            local_only=False, rpc_router=self.rpc_router, device=self.device
          )
      else:
        assert isinstance(self.dist_node_labels, Dict)
        if self.dist_node_labels is not None and \
            all(isinstance(value, Feature) for value in self.dist_node_labels.values()):
          self.dist_node_labels = DistFeature(
            self.data.num_partitions, self.data.partition_idx,
            self.data.node_labels, self.data.node_feat_pb,
            local_only=False, rpc_router=self.rpc_router, device=self.device
          )
    else:
      raise ValueError(f"'{self.__class__.__name__}': found invalid input "
                       f"data type '{type(data)}'")

    self.sampler = NeighborSampler(
      self.dist_graph.local_graph, self.num_neighbors,
      self.device, self.with_edge, self.with_neg, self.with_weight, 
      self.edge_dir, seed=self.seed
    )
    self.inducer_pool = queue.Queue(maxsize=self.concurrency)

    # rpc register
    rpc_sample_callee = RpcSamplingCallee(self.sampler, self.device)
    self.rpc_sample_callee_id = rpc_register(rpc_sample_callee)
    rpc_subgraph_callee = RpcSubGraphCallee(self.sampler, self.device)
    self.rpc_subgraph_callee_id = rpc_register(rpc_subgraph_callee)

    if self.dist_graph.data_cls == 'hetero':
      self.num_neighbors = self.sampler.num_neighbors
      self.num_hops = self.sampler.num_hops
      self.edge_types = self.sampler.edge_types

    super().__init__(self.concurrency)
    self._loop.call_soon_threadsafe(ensure_device, self.device)

  def sample_from_nodes(
    self,
    inputs: NodeSamplerInput,
    **kwargs
  ) -> Optional[SampleMessage]:
    r""" Sample multi-hop neighbors from nodes, collect the remote features
    (optional), and send results to the output channel.

    Note that if the output sample channel is specified, this func is
    asynchronized and the sampled result will not be returned directly.
    Otherwise, this func will be blocked to wait for the sampled result and
    return it.

    Args:
      inputs (NodeSamplerInput): The input data with node indices to start
        sampling from.
    """
    inputs = NodeSamplerInput.cast(inputs)
    if self.channel is None:
      return self.run_task(coro=self._send_adapter(self._sample_from_nodes,
                                                   inputs))
    cb = kwargs.get('callback', None)
    self.add_task(coro=self._send_adapter(self._sample_from_nodes, inputs),
                  callback=cb)
    return None

  def sample_from_edges(
    self,
    inputs: EdgeSamplerInput,
    **kwargs,
  ) -> Optional[SampleMessage]:
    r""" Sample multi-hop neighbors from edges, collect the remote features
    (optional), and send results to the output channel.

    Note that if the output sample channel is specified, this func is
    asynchronized and the sampled result will not be returned directly.
    Otherwise, this func will be blocked to wait for the sampled result and
    return it.

    Args:
      inputs (EdgeSamplerInput): The input data for sampling from edges
        including the (1) source node indices, the (2) destination node
        indices, the (3) optional edge labels and the (4) input edge type.
    """
    if self.channel is None:
      return self.run_task(coro=self._send_adapter(self._sample_from_edges,
                                                   inputs))
    cb = kwargs.get('callback', None)
    self.add_task(coro=self._send_adapter(self._sample_from_edges, inputs),
                  callback=cb)
    return None

  def subgraph(
    self,
    inputs: NodeSamplerInput,
    **kwargs
  ) -> Optional[SampleMessage]:
    r""" Induce an enclosing subgraph based on inputs and their neighbors(if
      self.num_neighbors is not None).
    """
    inputs = NodeSamplerInput.cast(inputs)
    if self.channel is None:
      return self.run_task(coro=self._send_adapter(self._subgraph, inputs))
    cb = kwargs.get('callback', None)
    self.add_task(coro=self._send_adapter(self._subgraph, inputs), callback=cb)
    return None

  async def _send_adapter(
    self,
    async_func,
    *args, **kwargs
  ) -> Optional[SampleMessage]:
    sampler_output = await async_func(*args, **kwargs)
    res = await self._colloate_fn(sampler_output)
    if self.channel is None:
      return res
    self.channel.send(res)
    return None

  async def _sample_from_nodes(
    self,
    inputs: NodeSamplerInput,
  ) -> Optional[SampleMessage]:
    input_seeds = inputs.node.to(self.device)
    input_type = inputs.input_type
    self.max_input_size = max(self.max_input_size, input_seeds.numel())
    inducer = self._acquire_inducer()
    is_hetero = (self.dist_graph.data_cls == 'hetero')
    if is_hetero:
      assert input_type is not None
      src_dict = inducer.init_node({input_type: input_seeds})
      batch = src_dict
      out_nodes, out_rows, out_cols, out_edges = {}, {}, {}, {}
      num_sampled_nodes, num_sampled_edges = {}, {}
      merge_dict(src_dict, out_nodes)
      count_dict(src_dict, num_sampled_nodes, 1)

      for i in range(self.num_hops):
        task_dict, nbr_dict, edge_dict = {}, {}, {}
        for etype in self.edge_types:
          req_num = self.num_neighbors[etype][i]
          if self.edge_dir == 'in':
            srcs = src_dict.get(etype[-1], None)
            if srcs is not None and srcs.numel() > 0:
              task_dict[reverse_edge_type(etype)] = self._loop.create_task(
                self._sample_one_hop(srcs, req_num, etype))
          elif self.edge_dir == 'out':
            srcs = src_dict.get(etype[0], None)
            if srcs is not None and srcs.numel() > 0:
              task_dict[etype] = self._loop.create_task(
                self._sample_one_hop(srcs, req_num, etype))

        for etype, task in task_dict.items():
          output: NeighborOutput = await task
          if output.nbr.numel() == 0:
            continue
          nbr_dict[etype] = [src_dict[etype[0]], output.nbr, output.nbr_num]
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

      sample_output = HeteroSamplerOutput(
        node={ntype: torch.cat(nodes) for ntype, nodes in out_nodes.items()},
        row={etype: torch.cat(rows) for etype, rows in out_rows.items()},
        col={etype: torch.cat(cols) for etype, cols in out_cols.items()},
        edge=(
          {etype: torch.cat(eids) for etype, eids in out_edges.items()}
          if self.with_edge else None
        ),
        batch=batch,
        num_sampled_nodes=num_sampled_nodes,
        num_sampled_edges=num_sampled_edges,
        input_type=input_type,
        metadata={}
      )

    else:
      srcs = inducer.init_node(input_seeds)
      batch = srcs
      out_nodes, out_edges = [], []
      num_sampled_nodes, num_sampled_edges = [], []
      out_nodes.append(srcs)
      num_sampled_nodes.append(srcs.size(0))
      # Sample subgraph.
      for req_num in self.num_neighbors:
        output: NeighborOutput = await self._sample_one_hop(srcs, req_num, None)
        if output.nbr.numel() == 0:
          break
        nodes, rows, cols = \
          inducer.induce_next(srcs, output.nbr, output.nbr_num)
        out_nodes.append(nodes)
        out_edges.append((rows, cols, output.edge))
        num_sampled_nodes.append(nodes.size(0))
        num_sampled_edges.append(cols.size(0))
        srcs = nodes

      sample_output = SamplerOutput(
        node=torch.cat(out_nodes),
        row=torch.cat([e[0] for e in out_edges]),
        col=torch.cat([e[1] for e in out_edges]),
        edge=(torch.cat([e[2] for e in out_edges]) if self.with_edge else None),
        batch=batch,
        num_sampled_nodes=num_sampled_nodes,
        num_sampled_edges=num_sampled_edges,
        metadata={}
      )
    # Reclaim inducer into pool.
    self.inducer_pool.put(inducer)

    return sample_output


  async def _sample_from_edges(
    self,
    inputs: EdgeSamplerInput,
  ) -> Optional[SampleMessage]:
    r"""Performs sampling from an edge sampler input, leveraging a sampling
    function of the same signature as `node_sample`.

    Currently, we support the out-edge sampling manner, so we reverse the
    direction of src and dst for the output so that features of the sampled
    nodes during training can be aggregated from k-hop to (k-1)-hop nodes.

    Note: Negative sampling is performed locally and unable to fetch positive
    edges from remote, so the negative sampling in the distributed case is
    currently non-strict for both binary and triplet manner.
    """
    src = inputs.row.to(self.device)
    dst = inputs.col.to(self.device)
    edge_label = None if inputs.label is None else inputs.label.to(self.device)
    input_type = inputs.input_type
    neg_sampling = inputs.neg_sampling

    num_pos = src.numel()
    num_neg = 0
    # Negative Sampling
    self.sampler.lazy_init_neg_sampler()
    if neg_sampling is not None:
      # When we are doing negative sampling, we append negative information
      # of nodes/edges to `src`, `dst`.
      # Later on, we can easily reconstruct what belongs to positive and
      # negative examples by slicing via `num_pos`.
      num_neg = math.ceil(num_pos * neg_sampling.amount)
      if neg_sampling.is_binary():
        # In the "binary" case, we randomly sample negative pairs of nodes.
        if input_type is not None:
          neg_pair = self.sampler._neg_sampler[input_type].sample(num_neg)
        else:
          neg_pair = self.sampler._neg_sampler.sample(num_neg)
        src_neg, dst_neg = neg_pair[0], neg_pair[1]
        src = torch.cat([src, src_neg], dim=0)
        dst = torch.cat([dst, dst_neg], dim=0)
        if edge_label is None:
            edge_label = torch.ones(num_pos, device=self.device)
        size = (src_neg.size()[0], ) + edge_label.size()[1:]
        edge_neg_label = edge_label.new_zeros(size)
        edge_label = torch.cat([edge_label, edge_neg_label])
      elif neg_sampling.is_triplet():
        assert num_neg % num_pos == 0
        if input_type is not None:
          neg_pair = self.sampler._neg_sampler[input_type].sample(num_neg, padding=True)
        else:
          neg_pair = self.sampler._neg_sampler.sample(num_neg, padding=True)
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
        temp_out.append(await self._sample_from_nodes(seeds))
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
          edge_label_index = torch.stack([inverse_src, inverse_dst], dim=0)
        else:
          edge_label_index = inverse_seed.view(2, -1)

        out.metadata.update({'edge_label_index': edge_label_index,
                             'edge_label': edge_label})
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

        out.metadata.update({'src_index': src_index,
                             'dst_pos_index': dst_pos_index,
                             'dst_neg_index': dst_neg_index})
        out.input_type = input_type
    else: #homo
      seed = torch.cat([src, dst], dim=0)
      seed, inverse_seed = seed.unique(return_inverse=True)
      out = await self._sample_from_nodes(NodeSamplerInput.cast(seed))

      # edge_label
      if neg_sampling is None or neg_sampling.is_binary():
        edge_label_index = inverse_seed.view(2, -1)

        out.metadata.update({'edge_label_index': edge_label_index,
                             'edge_label': edge_label})
      elif neg_sampling.is_triplet():
        src_index = inverse_seed[:num_pos]
        dst_pos_index = inverse_seed[num_pos:2 * num_pos]
        dst_neg_index = inverse_seed[2 * num_pos:]
        dst_neg_index = dst_neg_index.view(num_pos, -1).squeeze(-1)
        out.metadata.update({'src_index': src_index,
                             'dst_pos_index': dst_pos_index,
                             'dst_neg_index': dst_neg_index})

    return out


  async def _subgraph(
    self,
    inputs: NodeSamplerInput,
  ) -> Optional[SampleMessage]:
    inputs = NodeSamplerInput.cast(inputs)
    input_seeds = inputs.node.to(self.device)
    is_hetero = (self.dist_graph.data_cls == 'hetero')
    if is_hetero:
      raise NotImplementedError
    else:
      # neighbor sampling.
      if self.num_neighbors is not None:
        nodes = [input_seeds]
        for num in self.num_neighbors:
          nbr = await self._sample_one_hop(nodes[-1], num, None)
          nodes.append(torch.unique(nbr.nbr))
        nodes = torch.cat(nodes)
      else:
        nodes = input_seeds
      nodes, mapping = torch.unique(nodes, return_inverse=True)
      nid2idx = id2idx(nodes)
      # subgraph inducing.
      partition_ids = self.dist_graph.get_node_partitions(nodes)
      partition_ids = partition_ids.to(self.device)
      rows, cols, eids, futs = [], [], [], []
      for i in range(self.data.num_partitions):
        pidx = (self.data.partition_idx + i) % self.data.num_partitions
        p_ids = torch.masked_select(nodes, (partition_ids == pidx))
        if p_ids.shape[0] > 0:
          if pidx == self.data.partition_idx:
            subgraph = self.sampler.subgraph_op.node_subgraph(nodes, self.with_edge)
            # relabel row and col indices.
            rows.append(nid2idx[subgraph.nodes[subgraph.rows]])
            cols.append(nid2idx[subgraph.nodes[subgraph.cols]])
            if self.with_edge:
              eids.append(subgraph.eids.to(self.device))
          else:
            to_worker = self.rpc_router.get_to_worker(pidx)
            futs.append(rpc_request_async(to_worker,
                                          self.rpc_subgraph_callee_id,
                                          args=(nodes.cpu(),),
                                          kwargs={'with_edge': self.with_edge}))
      if not len(futs) == 0:
        res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
        for res_fut in res_fut_list:
          res_nodes, res_rows, res_cols, res_eids = res_fut.wait()
          res_nodes = res_nodes.to(self.device)
          rows.append(nid2idx[res_nodes[res_rows]])
          cols.append(nid2idx[res_nodes[res_cols]])
          if self.with_edge:
            eids.append(res_eids.to(self.device))

      sample_output = SamplerOutput(
        node=nodes,
        row=torch.cat(rows),
        col=torch.cat(cols),
        edge=torch.cat(eids) if self.with_edge else None,
        device=self.device,
        metadata={'mapping': mapping[:input_seeds.numel()]})

      return sample_output

  def _acquire_inducer(self):
    if self.inducer_pool.empty():
      return self.sampler.create_inducer(self.max_input_size)
    return self.inducer_pool.get()

  def _stitch_sample_results(
    self,
    input_seeds: torch.Tensor,
    results: List[PartialNeighborOutput]
  ) -> NeighborOutput:
    r""" Stitch partitioned neighbor outputs into a complete one.
    """
    idx_list = [r.index for r in results]
    nbrs_list = [r.output.nbr for r in results]
    nbrs_num_list = [r.output.nbr_num for r in results]
    eids_list = [r.output.edge for r in results] if self.with_edge else []
    if self.device.type == 'cuda':
      nbrs, nbrs_num, eids = pywrap.cuda_stitch_sample_results(
        input_seeds, idx_list, nbrs_list, nbrs_num_list, eids_list)
    else:
      nbrs, nbrs_num, eids = pywrap.cpu_stitch_sample_results(
        input_seeds, idx_list, nbrs_list, nbrs_num_list, eids_list)
    return NeighborOutput(nbrs, nbrs_num, eids)

  async def _sample_one_hop(
    self,
    srcs: torch.Tensor,
    num_nbr: int,
    etype: Optional[EdgeType]
  ) -> NeighborOutput:
    r""" Sample one-hop neighbors and induce the coo format subgraph.

    Args:
      srcs: input ids, 1D tensor.
      num_nbr: request(max) number of neighbors for one hop.
      etype: edge type to sample from input ids.

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: unique node ids and edge_index.
    """
    device = self.device
    srcs = srcs.to(device)
    if self.data.graph_caching:
      nbr_out = None
      if srcs is not None and srcs.numel() > 0:
        nbr_out = self.sampler.sample_one_hop(srcs, num_nbr, etype)
      return nbr_out

    orders = torch.arange(srcs.size(0), dtype=torch.long, device=device)
    if self.edge_dir == 'out':
      src_ntype = etype[0] if etype is not None else None
    elif self.edge_dir == 'in':
      src_ntype = etype[-1] if etype is not None else None
    partition_ids = self.dist_graph.get_node_partitions(srcs, src_ntype)
    partition_ids = partition_ids.to(device)
    partition_results: List[PartialNeighborOutput] = []
    remote_orders_list: List[torch.Tensor] = []
    futs: List[torch.futures.Future] = []

    for i in range(self.data.num_partitions):
      pidx = (
        (self.data.partition_idx + i) % self.data.num_partitions
      )
      p_mask = (partition_ids == pidx)
      if isinstance(self.dist_graph.node_pb, Dict):
        p_ids = self.data.id_select(srcs, p_mask, self.dist_graph.node_pb[src_ntype])
      else:
        p_ids = self.data.id_select(srcs, p_mask, self.dist_graph.node_pb)
      if p_ids.shape[0] > 0:
        p_orders = torch.masked_select(orders, p_mask)
        if pidx == self.data.partition_idx:
          p_nbr_out = self.sampler.sample_one_hop(p_ids, num_nbr, etype)
          partition_results.append(PartialNeighborOutput(p_orders, p_nbr_out))
        else:
          remote_orders_list.append(p_orders)
          to_worker = self.rpc_router.get_to_worker(pidx)
          futs.append(rpc_request_async(to_worker,
                                        self.rpc_sample_callee_id,
                                        args=(p_ids.cpu(), num_nbr, etype)))
    # Without remote sampling results.
    if len(remote_orders_list) == 0:
      if len(partition_results) > 0:
        return partition_results[0].output
      else:
        return torch.tensor([], device=self.device, dtype=torch.int64)
    # With remote sampling results.
    if not len(futs) == 0:
      res_fut_list = await wrap_torch_future(torch.futures.collect_all(futs))
      for i, res_fut in enumerate(res_fut_list):
        partition_results.append(
          PartialNeighborOutput(
            index=remote_orders_list[i],
            output=res_fut.wait().to(device)
          )
        )
    return self._stitch_sample_results(srcs, partition_results)

  async def _colloate_fn(
    self,
    output: Union[SamplerOutput, HeteroSamplerOutput]
  ) -> SampleMessage:
    r""" Collect labels and features for the sampled subgrarph if necessary,
    and put them into a sample message.
    """
    result_map = {}
    is_hetero = (self.dist_graph.data_cls == 'hetero')
    result_map['#IS_HETERO'] = torch.LongTensor([int(is_hetero)])
    if isinstance(output.metadata, dict):
      # scan kv and add metadata
      for k, v in output.metadata.items():
        result_map[f'#META.{k}'] = v

    if is_hetero:
      for ntype, nodes in output.node.items():
        result_map[f'{as_str(ntype)}.ids'] = nodes
        if output.num_sampled_nodes is not None:
          if ntype in output.num_sampled_nodes:
            result_map[f'{as_str(ntype)}.num_sampled_nodes'] = \
              torch.tensor(output.num_sampled_nodes[ntype], device=self.device)
      for etype, rows in output.row.items():
        etype_str = as_str(etype)
        result_map[f'{etype_str}.rows'] = rows
        result_map[f'{etype_str}.cols'] = output.col[etype]
        if self.with_edge:
          result_map[f'{etype_str}.eids'] = output.edge[etype]
        if output.num_sampled_edges is not None:
          if etype in output.num_sampled_edges:
            result_map[f'{etype_str}.num_sampled_edges'] = \
              torch.tensor(output.num_sampled_edges[etype], device=self.device)
      # Collect node labels of input node type.
      input_type = output.input_type
      assert input_type is not None
      if not isinstance(input_type, Tuple):
        if self.dist_node_labels is not None:
          if isinstance(self.dist_node_labels, DistFeature):
            fut = self.dist_node_labels.async_get(output.node[input_type], input_type)
            nlabels = await wrap_torch_future(fut)
            result_map[f'{as_str(input_type)}.nlabels'] = nlabels.T[0]
          else:
            node_labels = self.dist_node_labels.get(input_type, None)
            if node_labels is not None:
              result_map[f'{as_str(input_type)}.nlabels'] = \
                node_labels[output.node[input_type].to(node_labels.device)]
      # Collect node features.
      if self.dist_node_feature is not None:
        if self.use_all2all:
          sorted_ntype = sorted(self.dist_node_feature.feature_pb.keys())
          nfeat_dict = self.dist_node_feature.get_all2all(output, sorted_ntype)
          for ntype, nfeats in nfeat_dict.items():
            result_map[f'{as_str(ntype)}.nfeats'] = nfeats
        else:
          nfeat_fut_dict = {}
          for ntype, nodes in output.node.items():
            nodes = nodes.to(torch.long)
            nfeat_fut_dict[ntype] = self.dist_node_feature.async_get(nodes, ntype)
          for ntype, fut in nfeat_fut_dict.items():
            nfeats = await wrap_torch_future(fut)
            result_map[f'{as_str(ntype)}.nfeats'] = nfeats
      # Collect edge features
      if self.dist_edge_feature is not None and self.with_edge:
        efeat_fut_dict = {}
        for etype in self.edge_types:
          if self.edge_dir == 'out':
            eids = result_map.get(f'{as_str(etype)}.eids', None)
          elif self.edge_dir == 'in':
            eids = result_map.get(
              f'{as_str(reverse_edge_type(etype))}.eids', None)
          if eids is not None:
            eids = eids.to(torch.long)
            efeat_fut_dict[etype] = self.dist_edge_feature.async_get(eids, etype)
        for etype, fut in efeat_fut_dict.items():
          efeats = await wrap_torch_future(fut)
          if self.edge_dir == 'out':
            result_map[f'{as_str(etype)}.efeats'] = efeats
          elif self.edge_dir == 'in':
            result_map[f'{as_str(reverse_edge_type(etype))}.efeats'] = efeats
      # Collect batch info
      if output.batch is not None:
        for ntype, batch in output.batch.items():
          result_map[f'{as_str(ntype)}.batch'] = batch
    else:
      result_map['ids'] = output.node
      result_map['rows'] = output.row
      result_map['cols'] = output.col
      if output.num_sampled_nodes is not None:
        result_map['num_sampled_nodes'] = \
          torch.tensor(output.num_sampled_nodes, device=self.device)
        result_map['num_sampled_edges'] = \
          torch.tensor(output.num_sampled_edges, device=self.device)
      if self.with_edge:
        result_map['eids'] = output.edge
      # Collect node labels.
      if self.dist_node_labels is not None:
        if isinstance(self.dist_node_labels, DistFeature):
          fut = self.dist_node_labels.async_get(output.node)
          nlabels = await wrap_torch_future(fut)
          result_map['nlabels'] = nlabels.T[0]
        else:
          result_map['nlabels'] = \
            self.dist_node_labels[output.node.to(self.dist_node_labels.device)]
      # Collect node features.
      if self.dist_node_feature is not None:
        fut = self.dist_node_feature.async_get(output.node)
        nfeats = await wrap_torch_future(fut)
        result_map['nfeats'] = nfeats
      # Collect edge features.
      if self.dist_edge_feature is not None:
        eids = result_map['eids']
        fut = self.dist_edge_feature.async_get(eids)
        efeats = await wrap_torch_future(fut)
        result_map['efeats'] = efeats
      # Collect batch info
      if output.batch is not None:
        result_map['batch'] = output.batch

    return result_map
