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

#include <stdexcept>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "graphlearn_torch/include/common.cuh"
#include "graphlearn_torch/include/func_factory.h"
#include "graphlearn_torch/include/unified_tensor.cuh"

namespace graphlearn_torch {

#define CHECK_CPU(x) AT_ASSERTM(!x.device().is_cuda(), #x " must be CPU tensor")
#define CHECK_CUDA(x, y) AT_ASSERTM(x.device().is_cuda() \
  && (x.device().index() ==y), #x " must be CUDA tensor")

constexpr int32_t WARP_SIZE = 32;

__device__ int32_t GetDeviceId(const int64_t *offsets,
                               const int32_t device_count,
                               const int64_t index) {
  int32_t i = 1;
  for (; i < device_count; ++i) {
    if (index < offsets[i]) {
      return i - 1;
    }
  }
  return device_count - 1;
}

template <typename T>
__global__ void GatherTensorKernel(T** dev_ptrs,
                                   const int64_t* offsets,
                                   int32_t device_count,
                                   const int64_t* indices,
                                   int32_t indice_length,
                                   int32_t stride,
                                   T* res) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int thread_num = gridDim.x * blockDim.x;
  // each warp take charge of one-column feature copy
  unsigned int warp_id = tid / WARP_SIZE;
  unsigned int warp_num = thread_num / WARP_SIZE;
  unsigned int warp_tidx = tid % WARP_SIZE;

  int64_t dev_index = 0;
  int64_t dev_offset = 0;
  T* dev_ptr = nullptr;
  int64_t src_copy_start = 0;
  int64_t dst_copy_start = 0;
  auto col_idx = warp_tidx;

  while (warp_id < indice_length) {
    col_idx = warp_tidx;
    dev_index = GetDeviceId(offsets, device_count, indices[warp_id]);
    dev_ptr = dev_ptrs[dev_index];
    dev_offset = indices[warp_id] - offsets[dev_index];
    src_copy_start = dev_offset * stride;
    dst_copy_start = warp_id * stride;
    for (; col_idx < stride; col_idx += WARP_SIZE) {
      res[dst_copy_start + col_idx] = dev_ptr[src_copy_start + col_idx];
    }
    warp_id += warp_num;
  }
}

template <typename T>
void RunGatherTensor(cudaStream_t stream, void** dev_ptrs,
                     const int64_t* offsets, int32_t device_count,
                     const int64_t* indices, int32_t indice_length,
                     int32_t stride, void* res) {
  int block_size = 0;
  int grid_size = 0;
  cudaOccupancyMaxPotentialBlockSize(
    &grid_size, &block_size, GatherTensorKernel<T>);
  GatherTensorKernel<T><<<grid_size, block_size, 0, stream>>>(
    reinterpret_cast<T**>(dev_ptrs), offsets, device_count,
    indices, indice_length, stride, reinterpret_cast<T*>(res));
  CUDACheckError();
}

class RunGatherTensorFactory: public FunctionFactory<
  /* Key Type */
  torch::Dtype,
  /* Return Type */
  void,
  /* Argument Types */
  cudaStream_t, void**, const int64_t*, int32_t,
  const int64_t*, int32_t, int32_t, void*
>{};

namespace run_gather_tensor_registration {

#define __REGISTER(ScalarT, T) \
  auto __##ScalarT##_registry = RunGatherTensorFactory::Get().Register( \
    torch::Dtype::ScalarT, &RunGatherTensor<T>);

__REGISTER(Byte, uint8_t)
__REGISTER(Char, int8_t)
__REGISTER(Short, int16_t)
__REGISTER(Int, int)
__REGISTER(Long, int64_t)
__REGISTER(Half, at::Half)
__REGISTER(Float, float)
__REGISTER(Double, double)
__REGISTER(ComplexHalf, c10::complex<c10::Half>)
__REGISTER(ComplexFloat, c10::complex<float>)
__REGISTER(ComplexDouble, c10::complex<double>)
__REGISTER(Bool, bool)
__REGISTER(QInt8, c10::qint8)
__REGISTER(QUInt8, c10::quint8)
__REGISTER(QInt32, c10::qint32)
__REGISTER(BFloat16, at::BFloat16)
__REGISTER(QUInt4x2, c10::quint4x2)
__REGISTER(QUInt2x4, c10::quint2x4)

}  // namespace run_gather_tensor_registration

SharedTensor::SharedTensor() {}

SharedTensor::SharedTensor(int32_t device,
                           cudaIpcMemHandle_t mem_handle,
                           const std::vector<int64_t>& shape)
    : device_(device), mem_handle_(mem_handle), shape_(shape) {}

std::tuple<int32_t, cudaIpcMemHandle_t, std::vector<int64_t>>
SharedTensor::ShareCUDAIpc() {
  return std::make_tuple(device_, mem_handle_, shape_);
}

void SharedTensor::FromCUDAIpc(
  std::tuple<int32_t, cudaIpcMemHandle_t, std::vector<int64_t>> ipc_data) {
  device_ = std::get<0>(ipc_data);
  mem_handle_ = std::get<1>(ipc_data);
  shape_ = std::get<2>(ipc_data);
}

UnifiedTensor::UnifiedTensor(int32_t device, torch::Dtype dtype)
  : device_(device), dtype_(dtype), device_count_(0),
    inited_(false), registered_ptr_(nullptr) {
  tensor_offsets_.push_back(0);
}

UnifiedTensor::~UnifiedTensor() {
  if (registered_ptr_ != nullptr) {
    cudaHostUnregister(registered_ptr_);
  }
}

// Note that SharedTensor must be appended in the same order
// as InitFrom().
void UnifiedTensor::AppendSharedTensor(const SharedTensor& item) {
  cudaSetDevice(device_);
  if (!inited_) {
    shape_.resize(item.shape_.size());
    shape_[0] = 0;
    for (int32_t index = 1; index < shape_.size(); ++index) {
      shape_[index] = item.shape_[index];
    }
    inited_ = true;
  }

  if (item.device_ >= 0) {
    int32_t access_src_to_dst, access_dst_to_src;
    cudaDeviceCanAccessPeer(&access_src_to_dst, device_, item.device_);
    cudaDeviceCanAccessPeer(&access_dst_to_src, item.device_, device_);
    if (!((access_src_to_dst && access_dst_to_src) || device_ == item.device_)) {
      throw std::invalid_argument(
        "Peer access must be supported between SharedTensors.");
    }
  }

  tensor_shapes_.push_back(item.shape_);
  tensor_devices_.push_back(item.device_);
  tensor_offsets_.push_back(
    tensor_offsets_[tensor_offsets_.size() - 1] + item.shape_[0]);
  void *ptr = nullptr;
  cudaIpcOpenMemHandle(&ptr, item.mem_handle_, cudaIpcMemLazyEnablePeerAccess);
  tensor_ptrs_.push_back(ptr);
  shape_[0] += item.shape_[0];
  device_count_ += 1;
  CUDACheckError();
}

// Note that this function must be called after all SharedTensor added.
void UnifiedTensor::AppendCPUTensor(const torch::Tensor& tensor) {
  cudaSetDevice(device_);
  CHECK_CPU(tensor);
  CheckEq(tensor.dtype().toScalarType(), dtype_);
  if (!inited_) {
    shape_.resize(tensor.dim());
    shape_[0] = 0;
    auto tensor_sizes = tensor.sizes();
    for (int32_t index = 1; index < shape_.size(); ++index) {
      shape_[index] = tensor_sizes[index];
    }
    inited_ = true;
  }
  // TODO(baole): shape check.
  tensor_shapes_.push_back(tensor.sizes().vec());
  tensor_devices_.push_back(-1);
  tensor_offsets_.push_back(
    tensor_offsets_[tensor_offsets_.size() - 1] + tensor.sizes()[0]);
  void* ptr = nullptr;
  size_t data_size = tensor.nbytes();

  CUDARegisterByBlock(tensor.data_ptr(), data_size, cudaHostRegisterMapped);
  registered_ptr_ = tensor.data_ptr();
  CUDACheckError();
  cudaHostGetDevicePointer(&ptr, tensor.data_ptr(), 0);
  CUDACheckError();
  tensor_ptrs_.push_back(ptr);
  device_count_ += 1;
  shape_[0] += tensor.size(0);
}

void UnifiedTensor::Initp2p(const std::vector<int32_t>& devices){
  for (int32_t i = 0; i < devices.size(); ++i) {
    int src = devices[i];
    if (src < 0) {continue;}
    cudaSetDevice(src);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, src);

    if (!prop.unifiedAddressing) {
      printf("Device %d does not support unified addressing!\n", i);
      continue;
    }
    if (prop.computeMode != cudaComputeModeDefault) {
      printf("Device %d is in an unsupported compute mode for this sample\n",i);
      continue;
    }

    for (int32_t j = i + 1; j < devices.size(); ++j) {
      int dst = devices[j];
      if (dst < 0) {continue;}
      int access_src_to_dst = 0;
      int access_dst_to_src = 0;
      cudaDeviceCanAccessPeer(&access_src_to_dst, src, dst);
      cudaDeviceCanAccessPeer(&access_dst_to_src, dst, src);
      if (access_src_to_dst && access_dst_to_src) {
        cudaSetDevice(src);
        cudaDeviceEnablePeerAccess(dst, 0);
        CUDACheckError();
        cudaSetDevice(dst);
        cudaDeviceEnablePeerAccess(src, 0);
        CUDACheckError();
      } else {
        throw std::invalid_argument("Peer access must be supported.");
      }
    }
  }
}

void UnifiedTensor::InitFrom(const std::vector<torch::Tensor>& tensors,
                             const std::vector<int32_t>& tensor_devices) {
  assert(tensors.size()==tensor_devices.size());
  Initp2p(tensor_devices);
  device_count_ = tensor_devices.size();
  tensor_devices_ = tensor_devices;
  for (int32_t i = 0; i < tensors.size(); ++i) {
    auto& tensor = tensors[i];
    auto tensor_device = tensor_devices[i];
    CHECK_CPU(tensor);
    CheckEq(tensor.dtype().toScalarType(), dtype_);
    if (!inited_) {
      shape_.resize(tensor.dim());
      shape_[0] = 0;
      auto tensor_sizes = tensor.sizes();
      for (int32_t index = 1; index < shape_.size(); ++index) {
        shape_[index] = tensor_sizes[index];
      }
      inited_ = true;
    }
    // TODO(baole): shape check.
    tensor_shapes_.push_back(tensor.sizes().vec());
    tensor_offsets_.push_back(tensor_offsets_[tensor_offsets_.size() - 1] +
                              tensor.sizes()[0]);
    void* ptr = nullptr;
    size_t data_size = tensor.nbytes();
    if (tensor_device >= 0) {
      cudaSetDevice(tensor_device);
      cudaMalloc(&ptr, data_size);
      cudaMemcpy(ptr, tensor.data_ptr(), data_size, cudaMemcpyHostToDevice);
      cudaSetDevice(device_);
    } else { // cpu tensor, we use zero-copy.
      cudaSetDevice(device_);
      CUDARegisterByBlock(tensor.data_ptr(), data_size, cudaHostRegisterMapped);
      registered_ptr_ = tensor.data_ptr();
      cudaHostGetDevicePointer(&ptr, tensor.data_ptr(), 0);
    }
    tensor_ptrs_.push_back(ptr);
    shape_[0] += tensor.size(0);
  }
}

torch::Tensor UnifiedTensor::operator[](const torch::Tensor &indices) {
  int current_device = 0;
  cudaGetDevice(&current_device);
  CHECK_CUDA(indices, current_device);
  auto stream = ::at::cuda::getDefaultCUDAStream();
  ::at::cuda::CUDAStreamGuard guard(stream);

  std::vector<int64_t> res_shape(shape_);
  res_shape[0] = indices.numel();
  auto options = torch::TensorOptions()
                   .dtype(dtype_)
                   .device(torch::kCUDA, current_device);
  auto res = torch::empty(res_shape, options);
  CUDACheckError();

  auto val = GetDeviceData(current_device);
  auto data_buffer = std::get<0>(val);
  auto data_offset = std::get<1>(val);
  RunGatherTensorFactory::Get().Dispatch(
    /* key */
    dtype_,
    /* arguments */
    stream, data_buffer, data_offset, tensor_offsets_.size(),
    indices.data_ptr<int64_t>(), indices.numel(), Stride(0), res.data_ptr());
  return res;
}

std::tuple<void**, int64_t*> UnifiedTensor::GetDeviceData(int32_t device) {
  auto iter = device_data_map_.find(device);
  if (iter == device_data_map_.end()) {
    void** data_buffer;
    int64_t* data_offset;

    cudaMalloc((void***)&data_buffer, sizeof(void*) * device_count_);
    cudaMemcpy(data_buffer,
               &tensor_ptrs_[0],
               sizeof(void*) * tensor_ptrs_.size(),
               cudaMemcpyHostToDevice);
    CUDACheckError();

    cudaMalloc((void**)&data_offset, sizeof(int64_t) * tensor_offsets_.size());
    cudaMemcpy(data_offset,
               &tensor_offsets_[0],
               sizeof(int64_t) * tensor_offsets_.size(),
               cudaMemcpyHostToDevice);
    CUDACheckError();

    auto res = std::make_tuple(data_buffer, data_offset);
    device_data_map_.emplace(device, res);
    return res;
  }
  return iter->second;
}

std::vector<SharedTensor> UnifiedTensor::ShareCUDAIpc() {
  std::vector<SharedTensor> res;
  for (int32_t index = 0; index < tensor_ptrs_.size(); ++index) {
    if (tensor_devices_[index] >= 0) {
      cudaSetDevice(tensor_devices_[index]);
      cudaIpcMemHandle_t mem_handle;
      cudaIpcGetMemHandle(&mem_handle, tensor_ptrs_[index]);
      CUDACheckError();
      res.emplace_back(tensor_devices_[index],
                       mem_handle,
                       tensor_shapes_[index]);
    }
  }
  return res;
}

const std::vector<int64_t> UnifiedTensor::Shape() const {
  return shape_;
}

int32_t UnifiedTensor::Device() const {
  return device_;
}

int32_t UnifiedTensor::Size(int32_t dim) const {
  if (shape_.size() == 0) {
    return 0;
  }
  return shape_[dim];
}

int64_t UnifiedTensor::Stride(int32_t dim) const {
  int64_t res = 1;
  for (int32_t index = dim + 1; index < shape_.size(); ++index) {
    res *= shape_[index];
  }
  return res;
}

int64_t UnifiedTensor::Numel() const {
  int64_t res = 1;
  for (int32_t index = 0; index < shape_.size(); ++index) {
    res *= shape_[index];
  }
  return res;
}

} // namespace graphlearn_torch
