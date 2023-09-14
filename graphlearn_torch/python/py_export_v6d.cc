/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <torch/extension.h>

#include "graphlearn_torch/v6d/vineyard_utils.h"

namespace py = pybind11;

using namespace graphlearn_torch::vineyard_utils;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Python bindings for vineyard utils C++ frontend";
  m.def("vineyard_to_csr", &ToCSR);
  m.def("load_vertex_feature_from_vineyard", &LoadVertexFeatures);
  m.def("load_edge_feature_from_vineyard", &LoadEdgeFeatures);
}
