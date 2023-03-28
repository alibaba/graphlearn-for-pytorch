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

from .. import py_graphlearn_torch as pywrap


data_type = {'int32': pywrap.DataType.Int32,
             'int64': pywrap.DataType.Int64,
             'float32': pywrap.DataType.Float32,
             'float64': pywrap.DataType.Float64
            }

def vineyard_to_csr(sock, fid, elid=0, vlid=0, haseid=0):
  '''
    Wrap to_csr function to read graph from vineyard
    with return (indptr, indices, (Optional)edge_id)
  '''
  return pywrap.vineyard_to_csr(sock, fid, elid, vlid, haseid)


def load_vertex_feature_from_vineyard(sock, fid, vcols,
                                      vlid=0, dtype='float32'):
  '''
    Wrap load_vertex_feature_from_vineyard function to read vertex feature
    from vineyard
    dtype: torch data type
    return vertex_feature(torch.Tensor)
  '''
  return pywrap.load_vertex_feature_from_vineyard(sock, fid, vlid, vcols,
                                                  data_type[dtype])


def load_edge_feature_from_vineyard(sock, fid, ecols,
                                    elid=0, dtype='float32'):
  '''
    Wrap load_edge_feature_from_vineyard function to read edge feature
    from vineyard
    dtype: torch data type
    return edge_feature(torch.Tensor)
  '''
  return pywrap.load_edge_feature_from_vineyard(sock, fid, elid, ecols,
                                                data_type[dtype])
