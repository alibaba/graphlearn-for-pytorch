# Basic Graph Operations


We will introduce the basic concepts and graph operations of GLT.


## Graph and Feature

GLT describes the graph topologcial data in CSR or CSC
format by an instance of [`graphlearn_torch.data.graph.Topology`](graphlearn_torch.data.graph.Topology).
The [`graphlearn_torch.data.graph.Graph`](graphlearn_torch.data.graph.Graph) uses
[`graphlearn_torch.data.graph.Topology`](graphlearn_torch.data.graph.Topology) as input data and stores the data
into either CPU memory or GPU memory according to different `mode`.

Node features are described by an instance of
[`graphlearn_torch.data.feature.Feature`](graphlearn_torch.data.feature.Feature), which uses a PyTorch CPU tensor
as the input and stores the data into GPU, pinned or CPU memory according to
the arguments `split_ratio` and `device_group_list`. The `split_ratio`
indicates the ratio of feature data stored in GPU and pinned memory, and the
`device_group_list` describes the GPU groups with peer-to-peer accsess which
is used to place GPU part data.


We show a simple example of a graph with three nodes and four edges.
Each node contains two features, and there is at least one GPU available.
The graph is stored in pinned memory and feature data is stored in GPU
and pinned memory.

``` python
import torch
from graphlearn_torch.data import Topology, DeviceGroup, Feature, Graph
# graph topology:
# 0
# | \
# 1  2
# Graph is stored in pin memory.
edge_index = torch.tensor([[0, 0, 1, 2],
                            [1, 2, 0, 0]], dtype=torch.long)
x = torch.tensor([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]], dtype=torch.float)
csr_topo = Topology(edge_index, layout='CSR')
graph = Graph(csr_topo, mode='ZERO_COPY', device=0)
# The 20% feature data is stored in GPU 0 and the remaining 80% is in
# pinned memory.
feature = Feature(x,
  split_ratio=0.2, device_group_list=[DeviceGroup(0, [0])], device=0)
```

## Neighbor Sampling

Sampling is essiential for large-scale graphs training.
GLT uses an instance of
[`graphlearn_torch.sampler.neighbor_sampler.NeighborSampler`](graphlearn_torch.sampler.neighbor_sampler.NeighborSampler)
as neighbor sampler which is similiar to PyG's neighbor sampler with the same input and output.

We show a 1-hop neighbor sampler example which samples 2 neighborhood nodes for
each input node.

``` python
import torch
from graphlearn_torch.data import Topology, Graph
from graphlearn_torch.sampler import NeighborSampler
# 0
# | \
# 1  2
edge_index = torch.tensor([[0, 0, 1, 2],
                           [1, 2, 0, 0]], dtype=torch.long)
csr_topo = Topology(edge_index, layout='CSR')
graph = Graph(csr_topo, mode='ZERO_COPY', device=0)
sampler = NeighborSampler(graph, [2])
input_seeds = torch.tensor([0, 1, 2], dtype=torch.long)
sample_out = sampler.sample_from_nodes(input_seeds)
print(sample_out.node, sample_out.row, sample_out.col)

>>> tensor([0, 1, 2], device='cuda:0')
    tensor([1, 2, 0, 0], device='cuda:0')
    tensor([0, 0, 1, 2], device='cuda:0')
```

## Dataset and DataLoader

To unify the management of graph and feature data, we define
a class [`graphlearn_torch.data.dataset.Dataset`](graphlearn_torch.data.dataset.Dataset).
To be compatible with PyG/PyTorch training, we also define a class [`graphlearn_torch.loader.neighbor_loader.NeighborLoader`](graphlearn_torch.loader.neighbor_loader.NeighborLoader), which is
similiar to PyG's `NeighborLoader`.


```python
import torch
from graphlearn_torch.data import Dataset
from graphlearn_torch.loader import NeighborLoader
# 0
# | \
# 1  2
edge_index = torch.tensor([[0, 0, 1, 2],
                            [1, 2, 0, 0]], dtype=torch.long)
x = torch.tensor([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]], dtype=torch.float)
dataset = Dataset()
dataset.init_graph(edge_index=edge_index)
dataset.init_node_features(node_feature_data=x, split_ratio=0.2,
                           device_group_list=[DeviceGroup(0, [0])])
input_seeds = torch.tensor([0, 1, 2], dtype=torch.long)
loader = NeighborLoader(dataset, [2], input_seeds)
for data in loader:
  print(data.x, data.edge_index)

>>> tensor([[0.0000, 1.0000],
            [0.1000, 1.1000],
            [0.2000, 1.2000]], device='cuda:0') tensor([[1, 2],
            [0, 0]], device='cuda:0')
    tensor([[0.1000, 1.1000],
            [0.0000, 1.0000]], device='cuda:0') tensor([[1],
            [0]], device='cuda:0') tensor([1, 0], device='cuda:0')
    tensor([[0.2000, 1.2000],
            [0.0000, 1.0000]], device='cuda:0') tensor([[1],
            [0]], device='cuda:0') tensor([2, 0], device='cuda:0')
```

## Training GNNs

Because GLT provides several dataloaders similar to PyG,
the output format of these dataloaders is the same as PyG's,
so the model part can use PyG's model directly.
We provide examples of [node classification](node_class) and
[link prediction](link_pred) based on PyG.
We also provide a simple [distributed training example](dist_train).
The whole examples can be found in GLT's github `exampes/` directorys.