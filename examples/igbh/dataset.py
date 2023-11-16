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

import numpy as np
import torch
import os.path as osp

import graphlearn_torch as glt

from torch_geometric.utils import add_self_loops, remove_self_loops
from download import download_dataset
from typing import Literal

class IGBHeteroDataset(object):
  def __init__(self,
               path,
               dataset_size='tiny',
               in_memory=True,
               use_label_2K=False,
               with_edges=True,
               layout: Literal['CSC', 'CSR', 'COO'] = 'COO',):
    self.dir = path
    self.dataset_size = dataset_size
    self.in_memory = in_memory
    self.use_label_2K = use_label_2K
    self.with_edges = with_edges
    self.layout = layout

    self.ntypes = ['paper', 'author', 'institute', 'fos']
    self.etypes = None
    self.edge_dict = {}
    self.feat_dict = {}
    self.paper_nodes_num = {'tiny':100000, 'small':1000000, 'medium':10000000, 'large':100000000, 'full':269346174}
    self.author_nodes_num = {'tiny':357041, 'small':1926066, 'medium':15544654, 'large':116959896, 'full':277220883}
    # 'paper' nodes.
    self.label = None
    self.train_idx = None
    self.val_idx = None
    self.test_idx = None
    if not osp.exists(osp.join(path, self.dataset_size, 'processed')):
      download_dataset(path, 'heterogeneous', dataset_size)
    self.process()

  def process(self):
    if self.with_edges:
      if self.layout == 'COO':
        if self.in_memory:
          paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
          'paper__cites__paper', 'edge_index.npy'))).t()
          author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
          'paper__written_by__author', 'edge_index.npy'))).t()
          affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
          'author__affiliated_to__institute', 'edge_index.npy'))).t()
          paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
          'paper__topic__fos', 'edge_index.npy'))).t()
          if self.dataset_size in ['large', 'full']:
            paper_published_journal = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
            'paper__published__journal', 'edge_index.npy'))).t()
            paper_venue_conference = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
            'paper__venue__conference', 'edge_index.npy'))).t()
        else:
          paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
          'paper__cites__paper', 'edge_index.npy'), mmap_mode='r')).t()
          author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
          'paper__written_by__author', 'edge_index.npy'), mmap_mode='r')).t()
          affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
          'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r')).t()
          paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
          'paper__topic__fos', 'edge_index.npy'), mmap_mode='r')).t()
          if self.dataset_size in ['large', 'full']:
            paper_published_journal = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
            'paper__published__journal', 'edge_index.npy'), mmap_mode='r')).t()
            paper_venue_conference = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
            'paper__venue__conference', 'edge_index.npy'), mmap_mode='r')).t()

        cites_edge = add_self_loops(remove_self_loops(paper_paper_edges)[0])[0]
        self.edge_dict = {
            ('paper', 'cites', 'paper'): (torch.cat([cites_edge[1, :], cites_edge[0, :]]), torch.cat([cites_edge[0, :], cites_edge[1, :]])),
            ('paper', 'written_by', 'author'): author_paper_edges,
            ('author', 'affiliated_to', 'institute'): affiliation_author_edges,
            ('paper', 'topic', 'fos'): paper_fos_edges,
            ('author', 'rev_written_by', 'paper'): (author_paper_edges[1, :], author_paper_edges[0, :]),
            ('institute', 'rev_affiliated_to', 'author'): (affiliation_author_edges[1, :], affiliation_author_edges[0, :]),
            ('fos', 'rev_topic', 'paper'): (paper_fos_edges[1, :], paper_fos_edges[0, :])
        }
        if self.dataset_size in ['large', 'full']:
          self.edge_dict[('paper', 'published', 'journal')] = paper_published_journal
          self.edge_dict[('paper', 'venue', 'conference')] = paper_venue_conference
          self.edge_dict[('journal', 'rev_published', 'paper')] = (paper_published_journal[1, :], paper_published_journal[0, :])
          self.edge_dict[('conference', 'rev_venue', 'paper')] = (paper_venue_conference[1, :], paper_venue_conference[0, :])
      
      # directly load from CSC or CSC files, which can be generated using compress_graph.py
      else:
        compress_edge_dict = {}
        compress_edge_dict[('paper', 'cites', 'paper')] = 'paper__cites__paper'
        compress_edge_dict[('paper', 'written_by', 'author')] = 'paper__written_by__author'
        compress_edge_dict[('author', 'affiliated_to', 'institute')] = 'author__affiliated_to__institute'
        compress_edge_dict[('paper', 'topic', 'fos')] = 'paper__topic__fos'
        compress_edge_dict[('author', 'rev_written_by', 'paper')] = 'author__rev_written_by__paper'
        compress_edge_dict[('institute', 'rev_affiliated_to', 'author')] = 'institute__rev_affiliated_to__author'
        compress_edge_dict[('fos', 'rev_topic', 'paper')] = 'fos__rev_topic__paper'
        if self.dataset_size in ['large', 'full']:
          compress_edge_dict[('paper', 'published', 'journal')] = 'paper__published__journal'
          compress_edge_dict[('paper', 'venue', 'conference')] = 'paper__venue__conference'
          compress_edge_dict[('journal', 'rev_published', 'paper')] = 'journal__rev_published__paper'
          compress_edge_dict[('conference', 'rev_venue', 'paper')] = 'conference__rev_venue__paper'
        
        for etype in compress_edge_dict.keys():
          edge_path = osp.join(self.dir, self.dataset_size, 'processed', self.layout, compress_edge_dict[etype])
          try:
            edge_path = osp.join(self.dir, self.dataset_size, 'processed', self.layout, compress_edge_dict[etype])
            indptr = torch.load(osp.join(edge_path, 'indptr.pt'))
            indices = torch.load(osp.join(edge_path, 'indices.pt'))
            if self.layout == 'CSC':
              self.edge_dict[etype] = (indices, indptr)
            else:
              self.edge_dict[etype] = (indptr, indices)
          except FileNotFoundError:
            print(f"FileNotFound: {file_path}")
            exit()
          except Exception as e:
            print(f"Exception: {e}")
            exit()

      self.etypes = list(self.edge_dict.keys())

    label_file = 'node_label_19.npy' if not self.use_label_2K else 'node_label_2K.npy'
    paper_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', 'node_feat.npy')
    paper_lbl_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', label_file)
    num_paper_nodes = self.paper_nodes_num[self.dataset_size]
    if self.in_memory:
      paper_node_features = torch.from_numpy(np.load(paper_feat_path))
      paper_node_labels = torch.from_numpy(np.load(paper_lbl_path)).to(torch.long) 
    else:
      if self.dataset_size in ['large', 'full']:
        paper_node_features = torch.from_numpy(np.memmap(paper_feat_path, dtype='float32', mode='r', shape=(num_paper_nodes,1024)))
        paper_node_labels = torch.from_numpy(np.memmap(paper_lbl_path, dtype='float32', mode='r', shape=(num_paper_nodes))).to(torch.long)
      else:
        paper_node_features = torch.from_numpy(np.load(paper_feat_path, mmap_mode='r'))
        paper_node_labels = torch.from_numpy(np.load(paper_lbl_path, mmap_mode='r')).to(torch.long)
    self.feat_dict['paper'] = paper_node_features
    self.label = paper_node_labels

    num_author_nodes = self.author_nodes_num[self.dataset_size]
    author_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'author', 'node_feat.npy')
    if self.in_memory:
      author_node_features = torch.from_numpy(np.load(author_feat_path))
    else:
      if self.dataset_size in ['large', 'full']:
        author_node_features = torch.from_numpy(np.memmap(author_feat_path, dtype='float32', mode='r', shape=(num_author_nodes,1024)))
      else:
        author_node_features = torch.from_numpy(np.load(author_feat_path, mmap_mode='r'))
    self.feat_dict['author'] = author_node_features

    if self.in_memory:
      institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'institute', 'node_feat.npy')))
    else:
      institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'institute', 'node_feat.npy'), mmap_mode='r'))
    self.feat_dict['institute'] = institute_node_features

    if self.in_memory:
      fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'fos', 'node_feat.npy')))
    else:
      fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'fos', 'node_feat.npy'), mmap_mode='r'))
    self.feat_dict['fos'] = fos_node_features

    if self.dataset_size in ['large', 'full']:
      if self.in_memory:
        conference_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'conference', 'node_feat.npy')))
      else:
        conference_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'conference', 'node_feat.npy'), mmap_mode='r'))
      self.feat_dict['conference'] = conference_node_features
      if self.in_memory:
        journal_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'journal', 'node_feat.npy')))
      else:
        journal_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'journal', 'node_feat.npy'), mmap_mode='r'))
      self.feat_dict['journal'] = journal_node_features

    n_labeled_idx = self.paper_nodes_num[self.dataset_size]
    if self.dataset_size == 'full':
      if self.use_label_2K:
          n_labeled_idx = 157675969
      else:
          n_labeled_idx = 227130858

    shuffled_index = torch.randperm(n_labeled_idx)
    n_train = int(n_labeled_idx * 0.6)
    n_val = int(n_labeled_idx * 0.2)

    self.train_idx = shuffled_index[:n_train]
    self.val_idx = shuffled_index[n_train : n_train + n_val]
    self.test_idx = shuffled_index[n_train + n_val:]
