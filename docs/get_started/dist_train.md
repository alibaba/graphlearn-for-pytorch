# Distributed Training

GLT supports efficient distributed training across multiple
GPUs or machines. Considering that GNN training is essentially
an asynchronous producer (sampling) consumer (training) model, the optimization
objective of the training process is the overall training throughput.
Correspondingly, the architecture of GLT can be divided into sampling
(sampler) and training (trainer) modules in general. GLT supports
two types of deployments in the distributed setting. In the **Worker Mode**,
the sampler and trainer are deployed on the same physical node (each node has
sampler & trainer at the same time). In the **Server-Client Mode**, the sampler
and trainer are separately deployed on different sets of physical nodes in
the cluster. Refer [Distributed Training Tutorial](../tutorial/dist.md) for
a more detailed introduction of distributed training GLT. Next, we use a
simple example to demonstrate distributed training using GLT in the worker mode.


## Dataset Preprocessing & Partitioning

We use the [OGBN-Products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) dataset
in this example. In distributed training (under the worker mode), each node in the cluster
holds a partition of the graph. Thus before the training starts, we partition the OGBN-Products
dataset into multiple partitions, each of which corresponds to a specific training worker.
GLT provides a preprocessing script for partitioning ogbn datasets.

Run the following scripts to partition OGBN-Products into 2 partitions.

```bash
cd examples/distributed
python partition_ogbn_dataset.py --dataset=ogbn-products --num_partitions=2
```

GLT supports caching graph topology and frequently accessed features
in GPU to accelerate GPU sampling and feature collection. For feature cache, we adopt a
pre-sampling-based approach to determine the hotness of vertices, and cache features for
vertices with higher hotness while loading the graph. The uncached feature data are stored in
pinned memory for efficient access via UVA. The above script also calculates
the hotness distribution of vertices and selects target vertices for feature cache. The cache capacity
is decided by a user-specified cache ratio. To partition your own dataset for distributed training,
please reference the ``Preprocessing and Data Partitioning`` section in this [tutorial](../tutorial/dist.md).

## Graph Loading
Once the graph partitioning is done, we can start loading the graph data from the
partitioned files for each node.

```python
print('--- Loading data partition ...')
root_dir = osp.join(osp.dirname(osp.realpath(__file__)), args.ogbn_products_root_dir)
data_pidx = args.node_rank % args.num_dataset_partitions
dataset = glt.distributed.DistDataset()
dataset.load(
  graph_dir=osp.join(root_dir, 'ogbn-products-partitions'),
  partition_idx=data_pidx,
  graph_mode='ZERO_COPY',
  whole_node_label_file=osp.join(root_dir, 'ogbn-products-label', 'label.pt')
)
train_idx = torch.load(
  osp.join(root_dir, 'ogbn-products-train-partitions', f'partition{data_pidx}.pt')
)

train_idx.share_memory_()
```

The [`graphlearn_torch.distributed.DistDataset`](graphlearn_torch.distributed.DistDataset)
manages the in-memory storage of partitioned graph topology and feature. The graph is loaded
into CPU/GPU memory by calling the ``load`` function, where ``partition_idx`` is the
index of the partition to be loaded, ``graph_mode`` determines where the topology is stored
(e.g., CPU memory or GPU memory) and the sampling devices (e.g., CPU or GPU). Note that, when the
graph topology is stored in CPU memory (not pinned), the sampling can only be performed by CPU. GPU sampling
must be applied when the graph is stored in the pinned CPU memory or GPU memory. The ``ZERO_COPY`` mode means
the graph topology is stored in pinned memory and can be accessed via UVA. We also load the ``train_idx`` for this partition. As multiple processes are involved in sampling and training, we move
``train_idx`` to shared memory.


## Training Process

The pipeline of model training is wrapped as function ``run_training_proc``, which is executed by
the training processes spawned in each node. This function consists of the following steps:

#### 1. Initialize the distributed worker group context for GLT.

```python
glt.distributed.init_worker_group(
  world_size=num_nodes*num_training_procs_per_node,
  rank=node_rank*num_training_procs_per_node+local_proc_rank,
  group_name='distributed-sage-supervised-trainer'
)

current_ctx = glt.distributed.get_context()
current_device = torch.device(local_proc_rank % torch.cuda.device_count())
```
By calling ``init_worker_group``, we pass the world size, the id of the current rank and
the name of the worker group to each training processing.
A [`graphlearn_torch.distributed.DistContext`](graphlearn_torch.distributed.DistContext)
object is created during the initialization of worker group, which stores the meta data of
the training process and its belonging worker group.

#### 2. Initialize the training process group of PyTorch.

```python
torch.distributed.init_process_group(
backend='nccl',
rank=current_ctx.rank,
world_size=current_ctx.world_size,
init_method='tcp://{}:{}'.format(master_addr, training_pg_master_port)
)
```

#### 3. Create distributed neighbor loader for training

```python
train_idx = train_idx.split(train_idx.size(0) // num_training_procs_per_node)[local_proc_rank]
train_loader = glt.distributed.DistNeighborLoader(
  data=dataset,
  num_neighbors=[15, 10, 5],
  input_nodes=train_idx,
  batch_size=batch_size,
  shuffle=True,
  collect_features=True,
  to_device=current_device,
  worker_options=glt.distributed.MpDistSamplingWorkerOptions(
    num_workers=2,
    worker_devices=[torch.device('cuda', i % torch.cuda.device_count() for i in range(2))]
    worker_concurrency=4,
    master_addr=master_addr,
    master_port=train_loader_master_port,
    channel_size='1GB',
    pin_memory=True
  )
)
```
Each training process has a [`graphlearn_torch.distributed.DistNeighborLoader`](graphlearn_torch.distributed.DistNeighborLoader),
which handles distributed graph sampling and feature collection for this process.
The ``worker_options`` stores a set of sampling-related options, including the number of
sampling processes created for each training process, the devices where the samplings are
performed, the number of batches each sampling process can process concurrently,
RPC related options, and options required for creating the message channel between sampling and training processes.
We abstract two types of worker options to accommodate different scenarios of deployment:
* [`graphlearn_torch.distributed.MpDistSamplingWorkerOptions`](graphlearn_torch.distributed.MpDistSamplingWorkerOptions)
specifies how distributed samplers are created in the worker mode.
* [`graphlearn_torch.distributed.RemoteDistSamplingWorkerOptions`](graphlearn_torch.distributed.RemoteDistSamplingWorkerOptions)
specifies how distributed samplers are created in the server-client mode.



#### 4. Define model and optimizer.
```python
torch.cuda.set_device(current_device)
model = GraphSAGE(
  in_channels=100,
  hidden_channels=256,
  num_layers=3,
  out_channels=47,
).to(current_device)
model = DistributedDataParallel(model, device_ids=[current_device.index])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```
We use GraphSAGE in this example, and adopt PyTorch's Distributed Data Parallel (DDP) for distributed training.

#### 5. Train the model
```python
for epoch in range(0, epochs):
model.train()
start = time.time()
for batch in train_loader:
  optimizer.zero_grad()
  out = model(batch.x, batch.edge_index)[:batch.batch_size].log_softmax(dim=-1)
  loss = F.nll_loss(out, batch.y[:batch.batch_size])
  loss.backward()
  optimizer.step()
end = time.time()
torch.cuda.empty_cache() # empty cache for efficient GPU memory.
torch.distributed.barrier()
print(f'-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}')
```

## Run the example code
Run the example by executing the following scripts:

```bash
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_sage_supervised.py \
  --num_nodes=2 --node_rank=0 --master_addr=localhost

# node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_sage_supervised.py \
  --num_nodes=2 --node_rank=1 --master_addr=localhost
```

The complete code of the above example can be found in `examples/distributed/dist_train_sage_supervised.py`.
An example of distributed training in server-client mode can be found in `examples/distributed/dist_train_sage_supervised_with_server.py`.
