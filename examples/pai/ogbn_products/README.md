## Run on PAI with MaxCompute(ODPS) table as input.

### 1. Prepare files
```
cd ..
tar -zcvf train_products_sage.tar.gz ogbn_products/*.py requirements.txt
```


### 2. Input
```
  ogbn_products_node(id bigint, feature string)
  ogbn_products_edge(src_id bigint, dst_id bigint, e_id bigint)
  ogbn_products_train(id bigint, label bigint)
```
You can first generate text files use
```
python data_preprocess.py
```

and then use tunnel to upload text files to ODPS Tables
```
create table if not exists ogbn_products_node(id bigint, feature string);
tunnel u /{your_path}/graphlearn-for-pytorch/data/products/ogbn_products_node ogbn_products_node -fd "\t";
create table if not exists ogbn_products_edge(src_id bigint, dst_id bigint, e_id bigint);
tunnel u /{your_path}/graphlearn-for-pytorch/data/products/ogbn_products_edge ogbn_products_edge -fd "\t";

create table if not exists ogbn_products_train(id bigint, label bigint);
tunnel u /{your_path}/graphlearn-for-pytorch/data/products/ogbn_products_train ogbn_products_train -fd "\t";
```

Note: **The src_id and dst_id of the edge and the id of the node must correspond,
and the ids are required to be encoded consecutively starting from 0.**


### 3. PAI command
#### 3.1 Single node single GPU

```
pai -name pytorch112z -Dscript='file://{your_path}/graphlearn-for-pytorch/examples/pai/train_products_sage.tar.gz' -DentryFile='ogbn_products/train_products_sage.py' -Dtables="odps://{your_project}/tables/ogbn_products_node,odps://{your_project}/tables/ogbn_products_edge,odps://{your_project}/tables/ogbn_products_train" -Dcluster="{\"worker\":{\"gpu\":100}}" -DworkerCount=1 -DuserDefinedParameters='--split_ratio=0.2';
```

#### 3.2 Single node(machine) multi GPUs
1 node with 4 GPUs

```
pai -name pytorch112z -Dscript='file://{your_path}/graphlearn-for-pytorch/examples/pai/train_products_sage.tar.gz' -DentryFile='ogbn_products/train_products_sage.py' -Dtables="odps://{your_project}/tables/ogbn_products_node,odps://{your_project}/tables/ogbn_products_edge,odps://{your_project}/tables/ogbn_products_train" -Dcluster="{\"worker\":{\"gpu\":400}}" -DworkerCount=1 -DuserDefinedParameters='--split_ratio=0.2';
```

#### 3.3 Mulit-nodes mulit GPUs
2 nodes each with 2 GPUs

```
pai -name pytorch112z -Dscript='file:///{your_path}/graphlearn-for-pytorch/examples/pai/train_products_sage.tar.gz' -DentryFile='ogbn_products/dist_train_products_sage.py' -Dtables="odps://{your_project}/tables/ogbn_products_node,odps://{your_project}/tables/ogbn_products_edge,odps://{your_project}/tables/ogbn_products_train" -Dcluster="{\"worker\":{\"gpu\":200}}" -DworkerCount=2 -DuserDefinedParameters='--num_training_procs=2';
```