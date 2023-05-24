# User Guide on Aibaba Cloud

## 简介
本文档用示例说明如何在[阿里云](https://www.aliyun.com/)上使用GraphLearn-for-pytorch(**GLT**)来训练GNN模型。

## 环境准备
### 产品官网
打开[阿里云**机器学习PAI**产品控制台](https://pai.console.aliyun.com/)。

### 数据和代码准备
可提前[创建NAS资源](https://nasnext.console.aliyun.com/overview)，并在NAS中克隆**GLT**的[代码库](https://github.com/alibaba/graphlearn-for-pytorch.git)。也可在下文DSW和DLC任务运行过程中准备代码和数据，具体步骤在后续章节详细描述。

## DSW
在[机器学习PAI产品控制台](https://pai.console.aliyun.com/)左侧选择交互式模式（DSW），进入DSW。
使用DSW可运行**GLT**的单机单卡和单机多卡的训练任务。

### 1. 创建实例
点击“**创建实例**”，进行配置。

“**地域及可用区**”、“**所属工作空间**”、“**实例名称**”、“**是否仅自己可见**”参考[DSW文档](https://help.aliyun.com/document_detail/163684.html?spm=a2c4g.202278.0.0.2bc84a4c9fVFQU)。

“**资源组**”：可以选择CPU规格，也可以选择GPU规格。多卡GPU规格的配置下，GraphLearn-for-torch能获得更优的性能，建议选择GPU规格。

“**数据集**”：数据集创建和挂载为可选。数据集用于存储GraphLearn-for-Pytorch任务运行的输入图数据、运行过程中的日志。当输入数据需要在多个DSW的任务中反复使用时，需要根据系统提示创建数据集。建议创建数据集并进行挂载，下文将以挂载路径为"/mnt/data"示例。

“**选择镜像**”：选择“镜像URL”，填入 `graphlearn/graphlearn_for_pytorch:1.0.0-ubuntu20.04-py3.8-torch1.13-cuda11.6`。

> :warning:
注意，该镜像要求“资源组”为“**GPU**”规格，我们暂时未提供预装**GLT**的CPU镜像，如需要使用CPU规格，请预先准备CPU镜像，并根据[文档](https://github.com/alibaba/graphlearn-for-pytorch/tree/main#installation)提示安装依赖的库和CPU版本的**GLT**。


### 2. 运行任务
1. 安装wheel包
```python
pip install graphlearn_torch ogb
# 在notebook中运行：!pip install graphlearn_torch ogb
```

2. Clone代码（如挂载的数据集中已存在，则可忽略此步骤）
```python
git clone https://github.com/alibaba/graphlearn-for-pytorch.git
```

3. 运行单机任务
- [单机单卡示例](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/train_sage_ogbn_products.py)，可notebook运行，也可以在terminal中运行。 任务运行过程中会下载图数据并进行处理。
- [单机多卡示例](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/multi_gpu/train_sage_ogbn_papers100m.py)，仅可terminal运行。notebook不支持multiprocessing。 任务运行过程中会下载图数据并进行处理。

## DLC
在[机器学习PAI产品控制台](https://pai.console.aliyun.com/)左侧选择容器训练（DLC），进入DLC。
我们使用DLC演示**GLT**多机多卡的训练任务。多机训练时，GLT的输入数据需要提前进行分片，因此，我们需要提交两个DLC任务：

1. 训练数据准备
2. 运行GNN训练任务

### 1. 训练数据准备
训练使用相同数据时，此任务只需要运行一次。

#### 1.1. 任务参数配置

“**资源组**”：参考[DLC文档](https://help.aliyun.com/document_detail/202278.html?spm=5176.12818093.help.58.6fb616d0ijseHQ#task-2037310)

“**任务名称**”：自定义

“**节点镜像**”： 选择“镜像地址”，填写`graphlearn/graphlearn_for_pytorch:1.0.0-ubuntu20.04-py3.8-torch1.13-cuda11.6`

“**任务类型**”： PyTorch

“**数据集配置**”： 根据系统提示创建数据集并进行挂载，下文以挂载路径“/mnt/data”为例进行说明。

“**代码配置**”：可忽略

“**执行命令**”：

```python
pip install graphlearn-torch && pip install ogb &&
cd /mnt/data/code/ &&
git clone https://github.com/alibaba/graphlearn-for-pytorch.git &&
cd graphlearn-for-pytorch/examples/distributed &&
echo "y\ny" > input.txt &&
python partition_ogbn_dataset.py < input.txt --dataset=ogbn-products --num_partitions=2
```

说明：

- `git clone ...` :

克隆graphlearn-for-pytorch的代码，如果挂载的文件系统中已经存在，则不再需要运行。

- `echo "y\ny" > input.txt && python partition_ogbn_dataset.py < input.txt`:

`partition_ogbn_dataset.py`运行时会接受interactive的输入，询问用户是否下载数据，如数据已经存在可不再重复下载。由于DLC运行时无法interactive输入，我们将输入重定向为文件`input.txt`；
`echo -e "y\ny" | python XXX`的方式在DLC上不work。

> :warning:
注意，`--num_partitions`参数填写必须和后续“运行GNN训练任务”中的Worker数一致。

"**三方库配置**"：忽略

“**容错监控**”：可选，详细参考[DLC用户文档](https://help.aliyun.com/document_detail/202278.html?spm=5176.12818093.help.58.6fb616d0ijseHQ#task-2037310)

“**专有网络配置**”：可选，详细参考[DLC用户文档](https://help.aliyun.com/document_detail/202278.html?spm=5176.12818093.help.58.6fb616d0ijseHQ#task-2037310)

#### 1.2. 任务资源配置

“**节点数量**”：1

“**节点配置**”：GPU实例， 建议选择 `ecs.gn6v-c8g1.2xlarge`。

#### 1.3. 提交任务
点击“提交”，等待任务运行完成。此任务大概需要运行10分钟，受网络状态影响。

### 2. 运行GNN训练任务

#### 2.1. 任务参数配置

“**资源组**”：参考[DLC文档](https://help.aliyun.com/document_detail/202278.html?spm=5176.12818093.help.58.6fb616d0ijseHQ#task-2037310)

“**任务名称**”：自定义

“**节点镜像**”：选择“镜像地址”，填写`graphlearn/graphlearn_for_pytorch:1.0.0-ubuntu20.04-py3.8-torch1.13-cuda11.6`

“**任务类型**”： PyTorch

“**数据集配置**”： 配置NAS数据集，并挂载和**任务1中相同的路径**

“**代码配置**”：可忽略

“**执行命令**”：

```python
pip install graphlearn-torch && pip install ogb &&
cd /mnt/data/code/graphlearn-for-pytorch/examples/distributed &&
python dist_train_sage_supervised.py --master_addr=$MASTER_ADDR \
--training_pg_master_port=12345 \
--train_loader_master_port=12346 \
--test_loader_master_port=12347 \
--num_training_procs=1 \
--num_nodes=$WORLD_SIZE \
--node_rank=$RANK
```

参数说明详见[代码](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/examples/distributed/dist_train_sage_supervised.py#L161)。

#### 2.2. 任务资源配置

“**节点数量**”：2（此参数和“训练数据准备”中的“num_partitions”参数需一致。）

“**节点配置**”：GPU实例， 多机单卡可选择`ecs.gn6v-c8g1.2xlarge`，多机多卡可选择`ecs.gn6v-c8g1.8xlarge`。

#### 2.3. 提交任务
点击“提交”，等待任务运行完成。此任务多机单卡默认参数配置大概需要运行20分钟。


