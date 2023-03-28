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

import yaml
import argparse

import paramiko
import click

if __name__ == "__main__":
  parser = argparse.ArgumentParser('Run DistRandomSampler benchmarks.')
  parser.add_argument('--config', type=str, default='bench_dist_config.yml',
    help='paths to configuration file for benchmarks')
  parser.add_argument('--epochs', type=int, default=1,
    help='repeat epochs for sampling')
  parser.add_argument('--batch_size', type=int, default=2048,
    help='batch size for sampling')
  parser.add_argument('--shuffle', action="store_true",
    help='whether to shuffle input seeds at each epoch')
  parser.add_argument('--with_edge', action="store_true",
    help='whether to sample with edge ids')
  parser.add_argument('--collect_features', action='store_true',
    help='whether to collect features for sampled results')
  parser.add_argument('--worker_concurrency', type=int, default=4,
    help='concurrency for each sampling worker')
  parser.add_argument('--channel_size', type=str, default='4GB',
    help='memory used for shared-memory channel')
  parser.add_argument('--master_addr', type=str, default='0.0.0.0',
    help='master ip address for synchronization across all training nodes')
  parser.add_argument('--master_port', type=str, default='12345',
    help='port for synchronization across all training nodes')
  args = parser.parse_args()

  config = open(args.config, 'r')
  config = yaml.load(config)
  dataset = config['dataset']
  ip_list, port_list, username_list = config['nodes'], config['ports'], config['usernames']
  dst_path_list = config['dst_paths']
  node_ranks = config['node_ranks']
  num_nodes = len(node_ranks)
  visible_devices = config['visible_devices']
  python_bins = config['python_bins']
  num_cores = len(visible_devices[0].split(','))

  dataset_path = "../../data/"
  passwd_dict = {}
  for username, ip in zip(username_list, ip_list):
    passwd_dict[ip+username] = click.prompt('passwd for '+username+'@'+ip,
    hide_input=True)
  for username, ip, port, dst, noderk, device, pythonbin in zip(       
    username_list,
    ip_list,
    port_list,
    dst_path_list,
    node_ranks,
    visible_devices,
    python_bins,
  ):
    trans = paramiko.Transport((ip, port))
    trans.connect(username=username, password=passwd_dict[ip+username])
    ssh = paramiko.SSHClient()
    ssh._transport = trans

    to_bench_dir = 'cd '+dst+'/benchmarks/api/ '
    exec_bench = "tmux new -d 'CUDA_VISIBLE_DEVICES="+device+" "+pythonbin+" bench_dist_neighbor_loader.py --dataset="+dataset+" --node_rank="+str(noderk)+" --num_nodes="+str(num_nodes)+" --sample_nprocs="+str(num_cores)+" --master_addr="+args.master_addr+" --master_port="+args.master_port+ " --batch_size="+str(args.batch_size)+" --channel_size="+args.channel_size+" --epochs="+str(args.epochs)
    if args.collect_features:
      exec_bench += " --collect_features"
    if args.with_edge:
      exec_bench += " --with_edge"
    if args.shuffle:
      exec_bench += " --shuffle"

    print(to_bench_dir + ' && '+ exec_bench + " '")
    stdin, stdout, stderr = ssh.exec_command(to_bench_dir+' && '+exec_bench+" '", bufsize=1)
    print(stdout.read().decode())
    print(stderr.read().decode())
    ssh.close()
