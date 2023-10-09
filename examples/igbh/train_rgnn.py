# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
import argparse, datetime
import numpy as np
import os.path as osp
import sklearn.metrics
import time, tqdm
import torch
import warnings


import graphlearn_torch as glt

from dataset import IGBHeteroDataset
from rgnn import RGNN

torch.manual_seed(42)
warnings.filterwarnings("ignore")


def evaluate(model, dataloader):
  predictions = []
  labels = []
  with torch.no_grad():
    for batch in dataloader:
      batch_size = batch['paper'].batch_size
      out = model(batch.x_dict,
                  batch.edge_index_dict,
                  num_sampled_nodes_dict=batch.num_sampled_nodes,
                  num_sampled_edges_dict=batch.num_sampled_edges)[:batch_size]
      labels.append(batch['paper'].y[:batch_size].cpu().numpy())
      predictions.append(out.argmax(1).cpu().numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    acc = sklearn.metrics.accuracy_score(labels, predictions)
    return acc


def train(model, device, train_dataloader, val_dataloader, test_dataloader, args):
  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

  loss_fcn = torch.nn.CrossEntropyLoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  best_accuracy = 0
  training_start = time.time()
  for epoch in tqdm.tqdm(range(args.epochs)):
    model.train()
    total_loss = 0
    train_acc = 0
    idx = 0
    gpu_mem_alloc = 0
    epoch_start = time.time()
    for batch in train_dataloader:
      idx += 1
      batch_size = batch['paper'].batch_size
      out = model(batch.x_dict,
                  batch.edge_index_dict,
                  num_sampled_nodes_dict=batch.num_sampled_nodes,
                  num_sampled_edges_dict=batch.num_sampled_edges)[:batch_size]
      y = batch['paper'].y[:batch_size]
      loss = loss_fcn(out, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      train_acc += sklearn.metrics.accuracy_score(y.cpu().numpy(),
          out.argmax(1).detach().cpu().numpy())*100
      gpu_mem_alloc += (
          torch.cuda.max_memory_allocated() / 1000000
          if args.with_gpu
          else 0
      )
    train_acc /= idx
    gpu_mem_alloc /= idx

    if epoch%args.log_every == 0:
      model.eval()
      val_acc = evaluate(model, val_dataloader).item()*100
      if best_accuracy < val_acc:
        best_accuracy = val_acc

      tqdm.tqdm.write(
          "Epoch {:03d} | Loss {:.4f} | Train Acc {:.2f} | Val Acc {:.2f} | Time {} | GPU {:.1f} MB".format(
              epoch,
              total_loss,
              train_acc,
              val_acc,
              str(datetime.timedelta(seconds = int(time.time() - epoch_start))),
              gpu_mem_alloc
          )
      )

  model.eval()
  test_acc = evaluate(model, test_dataloader).item()*100
  print("Test Acc {:.2f}%".format(test_acc))
  print("Total time taken " + str(datetime.timedelta(seconds = int(time.time() - training_start))))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'data', 'igbh')
  glt.utils.ensure_dir(root)
  parser.add_argument('--path', type=str, default=root,
      help='path containing the datasets')
  parser.add_argument('--dataset_size', type=str, default='tiny',
      choices=['tiny', 'small', 'medium', 'large', 'full'],
      help='size of the datasets')
  parser.add_argument('--num_classes', type=int, default=2983,
      choices=[19, 2983], help='number of classes')
  parser.add_argument('--in_memory', type=int, default=0,
      choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
  # Model
  parser.add_argument('--model', type=str, default='rgat',
                      choices=['rgat', 'rsage'])
  # Model parameters
  parser.add_argument('--fan_out', type=str, default='15,10,5')
  parser.add_argument('--batch_size', type=int, default=5120)
  parser.add_argument('--hidden_channels', type=int, default=128)
  parser.add_argument('--learning_rate', type=int, default=0.01)
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--num_layers', type=int, default=3)
  parser.add_argument('--num_heads', type=int, default=4)
  parser.add_argument('--log_every', type=int, default=5)
  parser.add_argument("--cpu_mode", action="store_true",
      help="Only use CPU for sampling and training, default is False.")
  parser.add_argument("--edge_dir", type=str, default='in')
  parser.add_argument("--with_trim", action="store_true",
      help="use trim_to_layer function from pyG")
  args = parser.parse_args()
  args.with_gpu = (not args.cpu_mode) and torch.cuda.is_available()
  device = torch.device('cuda' if args.with_gpu else 'cpu')

  igbh_dataset = IGBHeteroDataset(args.path, args.dataset_size, args.in_memory,
                                  args.num_classes==2983)
  # init graphlearn_torch Dataset.
  glt_dataset = glt.data.Dataset(edge_dir=args.edge_dir)
  glt_dataset.init_graph(
    edge_index=igbh_dataset.edge_dict,
    graph_mode='ZERO_COPY' if args.with_gpu else 'CPU'
  )
  glt_dataset.init_node_features(
    node_feature_data=igbh_dataset.feat_dict,
    with_gpu=args.with_gpu,
    split_ratio=0.2,
    device_group_list=[glt.data.DeviceGroup(0, [0])]
  )
  glt_dataset.init_node_labels(node_label_data={'paper': igbh_dataset.label})

  train_loader = glt.loader.NeighborLoader(glt_dataset,
                                           [int(fanout) for fanout in args.fan_out.split(',')],
                                           ('paper', igbh_dataset.train_idx),
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=False,
                                           device=device)
  val_loader = glt.loader.NeighborLoader(glt_dataset,
                                         [int(fanout) for fanout in args.fan_out.split(',')],
                                         ('paper', igbh_dataset.val_idx),
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         drop_last=False,
                                         device=device)
  test_loader = glt.loader.NeighborLoader(glt_dataset,
                                          [int(fanout) for fanout in args.fan_out.split(',')],
                                          ('paper', igbh_dataset.test_idx),
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          drop_last=False,
                                          device=device)
  # model
  model = RGNN(igbh_dataset.etypes,
               igbh_dataset.feat_dict['paper'].shape[1],
               args.hidden_channels,
               args.num_classes,
               num_layers=args.num_layers,
               dropout=0.2,
               model=args.model,
               heads=args.num_heads,
               node_type='paper',
               with_trim=args.with_trim).to(device)
  train(model, device, train_loader, val_loader, test_loader, args)
