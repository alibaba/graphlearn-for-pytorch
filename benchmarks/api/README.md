## Benchmarks

### 1. Test on single node
1. neighbor sampling
```
python bench_sampler.py --backend=glt --sample_mode=GPU
```

2. feature lookup
```
python bench_feature.py --backend=glt --split_ratio=0.2
```

3. sampling and feature lookup
please refer to [examples/multi_gpu](../../examples/multi_gpu/)

Here, we present the scalability results of a single machine with A100 GPUs.
we use CUDA 11.4, PyTorch 1.12, and GLT 0.2.0rc2, and show the overall
throughput of neighbor sampling and feature lookup.

<p align="center">
  <img width="60%" src=../../docs/figures/scale_up.png />
</p>


### 2. Test on distributed settings.

Distributed neighbor loader (2 nodes each with 2 GPUs)

```
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python bench_dist_neighbor_loader.py --node_rank=0

# node 1:
CUDA_VISIBLE_DEVICES=2,3 python bench_dist_neighbor_loader.py --node_rank=1
```

**You can also use our script for the benchmark**:
#### Step 0: Setup a Distributed File System
**Note**: You may skip this step if you already set up folder(s) synchronized across machines.

To perform distributed sampling, files and codes need to be accessed across multiple machines. A distributed file system (i.e., NFS, Ceph, SSHFS, etc) exempt you from synchnonizing files such as partition information by hand.

For configuration details, please refer to the following links:

NFS: https://wiki.archlinux.org/index.php/NFS

SSHFS: https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh

Ceph: https://docs.ceph.com/en/latest/install/

#### Step 1: Prepare data
Here we use ogbn_products and partition it into 2 partitions.
```
python ../../examples/distributed/partition_ogbn_dataset.py --dataset=ogbn-products --num_partitions=2
```

#### Step 2: Set up the configure file
An example template for configure file is in the yaml format. e.g.  `bench_dist_config.yml`.

```
# IP addresses for all nodes.
nodes:
  - 0.0.0.0
  - 1.1.1.1

# ssh IP ports for each node.
ports: [22, 22]

# path to python with GLT envs
python_bins:
  - /path/to/python
  - /path/to/python
# username for remote IPs.
usernames:
  - root
  - root

# dataset name, e.g. products, papers.
dataset: products

node_ranks: [0, 1]

dst_paths:
  - /path/to/GLT
  - /path/to/GLT

# Setup visible cuda devices for each node.
visible_devices:
  - 0,1,2,3
  - 4,5,6,7
```

#### Step 3: Launch distributed jobs

```
pip install paramiko
pip install click
python run_dist_bench.py --config=bench_dist_config.yml --master_addr=0.0.0.0 --master_port=11234
```

Optional parameters which you can append after the command above includes:
```
--epochs: repeat epochs for sampling, (default=1)
--batch_size: batch size for sampling, (default=2048)
--shuffle: whether to shuffle input seeds at each epoch, (default=False)
--with_edge: whether to sample with edge ids, (default=False)
--collect_features: whether to collect features for sampled results, (default=False)
--worker_concurrency: concurrency for each sampling worker, (default=4)
--channel_size: memory used for shared-memory channel, (default='4GB')
```

Here, we present the scaling out results of distributed sampling and feature lookup.
We demonstrate the acceleration ratios of throughput for 2 nodes and 4 nodes,
compared to a single node with 2 A100 GPUs each node. The tests were carried o
ut using CUDA version 11.4, PyTorch version 1.12, and GLT version 0.2.0rc2.

<p align="center">
  <img width="60%" src=../../docs/figures/scale_out.png />
</p>
