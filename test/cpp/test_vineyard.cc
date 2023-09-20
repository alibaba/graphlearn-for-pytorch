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

#include "gtest/gtest.h"

#include "graphlearn_torch/v6d/vineyard_utils.h"

#include <grape/communication/communicator.h>
#include <grape/worker/comm_spec.h>

using namespace graphlearn_torch::vineyard_utils;

class VineyardTest : public ::testing::Test {
protected:
  void SetUp() override {
    ipc_socket = "/var/run/vineyard.sock";
    object_id_str = "29791161733259207";
    v_label_name = "person";
    e_label_name = "knows";
    vcols_1 = {"feat0"};
    vcols_2 = {"feat0", "feat1"};
    ecols_1 = {"feat0"};
    ecols_2 = {"feat0", "feat1"};
  }

protected:
  std::string ipc_socket;
  std::string object_id_str;
  std::string v_label_name;
  std::string e_label_name;
  std::vector<std::string> vcols_1;
  std::vector<std::string> vcols_2;
  std::vector<std::string> ecols_1;
  std::vector<std::string> ecols_2;
};

TEST_F(VineyardTest, VineyardTest) {

  grape::InitMPIComm();
  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);
    torch::Tensor out_indptr;
    torch::Tensor out_indices;
    torch::Tensor out_edge_ids;
    std::tie(out_indptr, out_indices, out_edge_ids) = ToCSR(ipc_socket, object_id_str, v_label_name, e_label_name, "out", true);
    // std::cout << "ToCSR out" << std::endl;
    // std::cout << "out_indptr" <<  out_indptr << std::endl;
    // std::cout << "out_indices" <<  out_indices << std::endl;
    // std::cout << "out_edge_ids" <<  out_edge_ids << std::endl;
    
    torch::Tensor in_indptr;
    torch::Tensor in_indices;
    torch::Tensor in_edge_ids;
    std::tie(in_indptr, in_indices, in_edge_ids) = ToCSR(ipc_socket, object_id_str, v_label_name, e_label_name, "in", true);
    // std::cout << "ToCSR in" << std::endl;
    // std::cout << "in_indptr" <<  in_indptr << std::endl;
    // std::cout << "in_indices" <<  in_indices << std::endl;
    // std::cout << "in_edge_ids" <<  in_edge_ids << std::endl;

    ASSERT_EQ(out_indptr.numel(), 5);
    ASSERT_EQ(out_indptr.dtype(), torch::kLong);
    ASSERT_TRUE(torch::allclose(out_indptr, torch::tensor({0, 0, 0, 0, 2}, torch::kLong)));
    
    ASSERT_EQ(out_indices.numel(), 2);
    ASSERT_EQ(out_indices.dtype(), torch::kLong);
    ASSERT_TRUE(torch::allclose(out_indices, torch::tensor({0, 2}, torch::kLong)));
    
    ASSERT_EQ(out_edge_ids.numel(), 2);
    ASSERT_EQ(out_edge_ids.dtype(), torch::kLong);
    ASSERT_TRUE(torch::allclose(out_edge_ids, torch::tensor({0, 1}, torch::kLong)));
    
    ASSERT_EQ(in_indptr.numel(), 5);
    ASSERT_EQ(in_indptr.dtype(), torch::kLong);
    ASSERT_TRUE(torch::allclose(in_indptr, torch::tensor({0, 1, 1, 2, 2}, torch::kLong)));
    
    ASSERT_EQ(in_indices.numel(), 2);
    ASSERT_EQ(in_indices.dtype(), torch::kLong);
    ASSERT_TRUE(torch::allclose(in_indices, torch::tensor({3, 3}, torch::kLong)));
    
    ASSERT_EQ(in_edge_ids.numel(), 2);
    ASSERT_EQ(in_edge_ids.dtype(), torch::kLong);
    ASSERT_TRUE(torch::allclose(in_edge_ids, torch::tensor({0, 1}, torch::kLong)));


    auto vfeat_1 = LoadVertexFeatures(ipc_socket, object_id_str, v_label_name, vcols_1);
    ASSERT_EQ(vfeat_1.sizes(), torch::IntArrayRef({4, 1}));
    ASSERT_EQ(vfeat_1.dtype(), torch::kDouble);

    auto efeat_1 = LoadEdgeFeatures(ipc_socket, object_id_str, e_label_name, ecols_1);
    ASSERT_EQ(efeat_1.sizes(), torch::IntArrayRef({2, 1}));
    ASSERT_EQ(efeat_1.dtype(), torch::kDouble);
    
    auto vfeat_2 = LoadVertexFeatures(ipc_socket, object_id_str, v_label_name, vcols_2);
    ASSERT_EQ(vfeat_2.sizes(), torch::IntArrayRef({4, 2}));
    ASSERT_EQ(vfeat_2.dtype(), torch::kDouble);

    auto efeat_2 = LoadEdgeFeatures(ipc_socket, object_id_str, e_label_name, ecols_2);
    ASSERT_EQ(efeat_2.sizes(), torch::IntArrayRef({2, 2}));
    ASSERT_EQ(efeat_2.dtype(), torch::kDouble);
  }
}