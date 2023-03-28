.. graphlearn_torch documentation master file, created by
   sphinx-quickstart on Mon Jan 30 07:33:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GraphLearn-for-PyTorch's documentation!
==================================================
**GraphLearn-for-PyTorch(GLT)** is a graph learning library for PyTorch that
makes distributed GNN training and inference easy and efficient.
It leverages the power of GPUs to accelerate graph sampling and utilizes
UVA to reduce the conversion and copying of features of vertices and edges.
For large-scale graphs, it supports distributed training on multiple GPUs or
multiple machines through fast distributed sampling and feature lookup.
Additionally, it provides flexible deployment for distributed training to meet
different requirements.

.. toctree::
   :maxdepth: 2
   :caption: Installation

   install/install

.. toctree::
   :maxdepth: 2
   :caption: Get Started

   get_started/graph_basic
   get_started/node_class
   get_started/link_pred
   get_started/hetero
   get_started/dist_train

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorial/basic_object
   tutorial/graph_ops
   tutorial/dist


.. toctree::
   :maxdepth: 3
   :caption: API Reference

   apis/modules

.. toctree::
   :maxdepth: 2
   :caption: Contribute to GLT

   contribute

.. toctree::
   :maxdepth: 2
   :caption: FAQ

   faq

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
