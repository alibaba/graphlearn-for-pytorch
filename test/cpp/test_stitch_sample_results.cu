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

#include "graphlearn_torch/include/stitch_sample_results.h"
#include "graphlearn_torch/include/common.cuh"

using namespace graphlearn_torch;

class StitchSampleResultsTest : public ::testing::Test {
protected:
  void Init(const torch::TensorOptions& options) {
    ids_ = torch::tensor({1, 2, 3, 4, 5, 6}, options);
    expect_nbrs_num_ = torch::tensor({0, 1, 2, 3, 4, 5}, options);
    expect_nbrs_ = torch::tensor({/* no nbrs for 1 */
                                  3, /* nbrs of 2 */
                                  4, 5, /* nbrs of 3 */
                                  5, 6, 7, /* nbrs of 4 */
                                  6, 7, 8, 9, /* nbrs of 5 */
                                  7, 8, 9, 10, 11 /* nbrs of 6 */}, options);
    expect_eids_ = torch::tensor({/* no nbrs for 1 */
                                  0,
                                  1,  2,
                                  3,  4,  5,
                                  6,  7,  8,  9,
                                  10, 11, 12, 13, 14}, options);

    idx_list_.clear();
    nbrs_list_.clear();
    nbrs_num_list_.clear();
    eids_list_.clear();
    // partition 0
    idx_list_.emplace_back(torch::tensor({0, 2}, options));
    nbrs_num_list_.emplace_back(torch::tensor({0, 2}, options));
    nbrs_list_.emplace_back(torch::tensor({4, 5}, options));
    eids_list_.emplace_back(torch::tensor({1, 2}, options));
    // partition 1
    idx_list_.emplace_back(torch::tensor({1, 5}, options));
    nbrs_num_list_.emplace_back(torch::tensor({1, 5}, options));
    nbrs_list_.emplace_back(torch::tensor({3, 7, 8, 9, 10, 11}, options));
    eids_list_.emplace_back(torch::tensor({0, 10, 11, 12, 13, 14}, options));
    // partition 2
    idx_list_.emplace_back(torch::tensor({3, 4}, options));
    nbrs_num_list_.emplace_back(torch::tensor({3, 4}, options));
    nbrs_list_.emplace_back(torch::tensor({5, 6, 7, 6, 7, 8, 9}, options));
    eids_list_.emplace_back(torch::tensor({3, 4, 5, 6, 7, 8, 9}, options));
  }

protected:
  torch::Tensor ids_;
  torch::Tensor expect_nbrs_;
  torch::Tensor expect_nbrs_num_;
  torch::Tensor expect_eids_;
  std::vector<torch::Tensor> idx_list_;
  std::vector<torch::Tensor> nbrs_list_;
  std::vector<torch::Tensor> nbrs_num_list_;
  std::vector<torch::Tensor> eids_list_;
};

TEST_F(StitchSampleResultsTest, CPUStitch) {
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  Init(options);
  auto res = CPUStitchSampleResults(ids_, idx_list_, nbrs_list_,
                                    nbrs_num_list_, {});
  EXPECT_TRUE(torch::equal(std::get<0>(res), expect_nbrs_));
  EXPECT_TRUE(torch::equal(std::get<1>(res), expect_nbrs_num_));
  EXPECT_EQ(std::get<2>(res).has_value(), false);
}

TEST_F(StitchSampleResultsTest, CUDAStitch) {
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, 0);
  Init(options);
  auto res = CUDAStitchSampleResults(ids_, idx_list_, nbrs_list_,
                                     nbrs_num_list_, eids_list_);
  CUDACheckError();
  EXPECT_TRUE(torch::equal(std::get<0>(res), expect_nbrs_));
  EXPECT_TRUE(torch::equal(std::get<1>(res), expect_nbrs_num_));
  EXPECT_TRUE(torch::equal(std::get<2>(res).value(), expect_eids_));
}
