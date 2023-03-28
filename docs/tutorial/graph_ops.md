# Graph Operators

Graph-Learn_torch(GLT) optimizes the end-to-end training throughput of GNN 
models by boosting the performance of graph sampling and feature collection. 
In GLT, we have implemented vertex-based
[`graphlearn_torch.sampler.NeighborSampler`](graphlearn_torch.sampler.NeighborSampler)
and 
[`graphlearn_torch.sampler.RandomNegativeSampler`](graphlearn_torch.sampler.RandomNegativeSampler).
Edge-based and subgraph-based samplers will be added in the next release.
In this tutorial, we introduce the detailed designs of these graph-related operators in GLT.

## NeighborSampler
Similar to PyG, GLT wraps the configuration, initialization and execution
of samplers inside data loaders. The following code illustrates an example
of [`graphlearn_torch.loader.NeighborLoader`](graphlearn_torch.loader.NeighborLoader) 
in single machine training.

``` python
# graphlearn_torch NeighborLoader
train_loader = glt.loader.NeighborLoader(glt_dataset,
                                         [15, 10, 5],
                                         split_idx['train'],
                                         batch_size=1024,
                                         shuffle=True,
                                         drop_last=True,
                                         device=device,
                                         as_pyg_v1=True)
```

During the initialization of the neighbor loader, an instance of 
[`graphlearn_torch.sampler.NeighborSampler`](graphlearn_torch.sampler.NeighborSampler)
is created.

``` python
class NeighborSampler(BaseSampler):
  r""" Neighbor Sampler.
  """
  def __init__(self,
               graph: Union[Graph, Dict[str, Graph]],
               num_neighbors: NumNeighbors,
               device: torch.device=torch.device('cuda', 0),
               with_edge: bool=False,
               strategy: str = 'random'):
```

To be compatible with PyG, ``NeighborSampler`` in GLT inherits the class[`torch_geometric.sampler.BaseSampler`](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/sampler/base.py)
in PyG. Both homogeneous and heterogeneous graphs are supported. 
The sampler instance is created with the user-specified parameters: 
the number of hops, number of neighbors of each hop and the device where
sampling operations are performed. GLT supports both CPU and GPU sampling. 
By setting ``with_edge`` to ``True``, edge ids are included in the sampled results.
Edges ids can be used to extract edge features. By default the sampling 
strategy is ``random sampling``.

We directly use the input and output formats of PyG sampler for the
``NeighborSampler`` in GLT. The input formats of 
[`graphlearn_torch.sampler.NeighborSampler`](graphlearn_torch.sampler.NeighborSampler)
is 
[`torch_geometric.sampler.NodeSamplerInput `](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/sampler/base.py)

```python
@dataclass
class NodeSamplerInput(CastMixin):
  r""" The sampling input of
  :meth:`~graphlearn_torch.sampler.BaseSampler.sample_from_nodes`.

  This class corresponds to :class:`~torch_geometric.sampler.NodeSamplerInput`:
  https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/sampler/base.py

  Args:
    node (torch.Tensor): The indices of seed nodes to start sampling from.
    input_type (str, optional): The input node type (in case of sampling in
      a heterogeneous graph). (default: :obj:`None`).
  """
  node: torch.Tensor
  input_type: Optional[NodeType] = None
  ```

The output format of sampling results on homogeneous graphs is
[torch_geometric.sampler.SamplerOutput](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/sampler/base.py).

```python
@dataclass
class SamplerOutput(CastMixin):
  r""" The sampling output of a :class:`~graphlearn_torch.sampler.BaseSampler` on
  homogeneous graphs.

  Args:
    node (torch.Tensor): The sampled nodes in the original graph.
    row (torch.Tensor): The source node indices of the sampled subgraph.
      Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
      corresponding to the nodes in the :obj:`node` tensor.
    col (torch.Tensor): The destination node indices of the sampled subgraph.
      Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
      corresponding to the nodes in the :obj:`node` tensor.
    edge (torch.Tensor, optional): The sampled edges in the original graph.
      This tensor is used to obtain edge features from the original
      graph. If no edge attributes are present, it may be omitted.
    batch (torch.Tensor, optional): The vector to identify the seed node
      for each sampled node. Can be present in case of disjoint subgraph
      sampling per seed node. (default: :obj:`None`).
    device (torch.device, optional): The device that all data of this output
      resides in. (default: :obj:`None`).
    metadata: (Any, optional): Additional metadata information.
      (default: :obj:`None`).
  """
  node: torch.Tensor
  row: torch.Tensor
  col: torch.Tensor
  edge: Optional[torch.Tensor]  = None
  batch: Optional[torch.Tensor] = None
  device: Optional[torch.device] = None
  metadata: Optional[Any] = None
  ```
The output format of sampling results on heterogeneous graphs is
[torch_geometric.sampler.HeteroSamplerOutput](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/sampler/base.py).

```python
@dataclass
class HeteroSamplerOutput(CastMixin):
  r""" The sampling output of a :class:`~graphlearn_torch.sampler.BaseSampler` on
  heterogeneous graphs.

  Args:
    node (Dict[str, torch.Tensor]): The sampled nodes in the original graph
      for each node type.
    row (Dict[Tuple[str, str, str], torch.Tensor]): The source node indices
      of the sampled subgraph for each edge type. Indices must be re-indexed
      to :obj:`{ 0, ..., num_nodes - 1 }` corresponding to the nodes in the
      :obj:`node` tensor of the source node type.
    col (Dict[Tuple[str, str, str], torch.Tensor]): The destination node
      indices of the sampled subgraph for each edge type. Indices must be
      re-indexed to :obj:`{ 0, ..., num_nodes - 1 }` corresponding to the nodes
      in the :obj:`node` tensor of the destination node type.
    edge (Dict[Tuple[str, str, str], torch.Tensor], optional): The sampled
      edges in the original graph for each edge type. This tensor is used to
      obtain edge features from the original graph. If no edge attributes are
      present, it may be omitted. (default: :obj:`None`).
    batch (Dict[str, torch.Tensor], optional): The vector to identify the
      seed node for each sampled node for each node type. Can be present
      in case of disjoint subgraph sampling per seed node.
      (default: :obj:`None`).
    edge_types: (List[Tuple[str, str, str]], optional): The list of edge types
      of the sampled subgraph. (default: :obj:`None`).
    device (torch.device, optional): The device that all data of this output
      resides in. (default: :obj:`None`).
    metadata: (Any, optional): Additional metadata information.
      (default: :obj:`None`)
  """
  node: Dict[NodeType, torch.Tensor]
  row: Dict[EdgeType, torch.Tensor]
  col: Dict[EdgeType, torch.Tensor]
  edge: Optional[Dict[EdgeType, torch.Tensor]] = None
  batch: Optional[Dict[NodeType, torch.Tensor]] = None
  edge_types: Optional[List[EdgeType]] = None
  device: Optional[torch.device] = None
  metadata: Optional[Any] = None
  ```

## Negative Sampler
GLT also supports GPU and CPU sampling for negative sampler. The below
code shows the [`graphlearn_torch.sampler.RandomNegativeSampler`](graphlearn_torch.sampler.RandomNegativeSampler) in GLT.

```python
class RandomNegativeSampler(object):
  r""" Random negative Sampler.

  Args:
    graph: A ``graphlearn_torch.data.Graph`` object.
    mode: Execution mode of sampling, 'CUDA' means sampling on
      GPU, 'CPU' means sampling on CPU.
  """
  def __init__(self, graph, mode='CUDA'):
    self._mode = mode
    if mode == 'CUDA':
      self._sampler = pywrap.CUDARandomNegativeSampler(graph.graph_handler)
    else:
      self._sampler = pywrap.CPURandomNegativeSampler(graph.graph_handler)
```

The inputs of the ``sample`` method in ``RandomNegativeSampler`` include:
 ``req_num``the number of maximum negative samples, ``trials_num`` the maximum number of 
 trails to generate enough negative samples, and ``padding`` specifies if the
 number of negative samples are smaller than ``req_num`` after ``trials_num``
 is used up, whether to use randomly generated samples (could be positive or negative samples) to complement the number of samples in the outputs.
 
```python
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
    rows, cols = self._sampler.sample(req_num, trials_num, padding)
    return torch.stack([rows, cols], dim=0)
```


