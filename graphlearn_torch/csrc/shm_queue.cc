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

#include "graphlearn_torch/include/shm_queue.h"

#include <sys/shm.h>

#include <cassert>
#include <cstring>
#include <thread>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "graphlearn_torch/include/common.cuh"
#else
#include "graphlearn_torch/include/common.h"
#endif
namespace graphlearn_torch {

ShmData::~ShmData() {
  if (meta_) {
    meta_->ReleaseBlock(block_id_);
  }
}

ShmData::ShmData(ShmData&& other) noexcept
    : data_(other.data_), len_(other.len_),
      block_id_(other.block_id_), meta_(std::move(other.meta_)) {
  other.data_ = nullptr;
  other.len_ = 0;
}

ShmData& ShmData::operator=(ShmData&& other) noexcept {
  if (this != &other) {
    data_ = other.data_;
    len_ = other.len_;
    block_id_ = other.block_id_;
    meta_ = std::move(other.meta_);
    other.data_ = nullptr;
    other.len_ = 0;
  }
  return *this;
}

void ShmQueueMeta::Initialize(size_t max_block_num, size_t max_buf_size) {
  max_block_num_ = max_block_num;
  max_buf_size_ = max_buf_size;
  block_meta_offset_ = sizeof(ShmQueueMeta);
  data_buf_offset_ = sizeof(ShmQueueMeta) + sizeof(BlockMeta) * max_block_num_;
  write_block_id_ = 0;
  read_block_id_ = 0;
  alloc_offset_ = 0;
  released_offset_ = 0;
  sem_init(&alloc_lock_, 1, 1);
  sem_init(&release_lock_, 1, 1);
  for (size_t i = 0; i < max_block_num_; i++) {
    GetBlockMeta(i).Initialize();
  }
}

void ShmQueueMeta::Finalize() {
  for (size_t i = 0; i < max_block_num_; i++) {
    GetBlockMeta(i).Finalize();
  }
  sem_destroy(&alloc_lock_);
  sem_destroy(&release_lock_);
}

size_t ShmQueueMeta::GetBlockToWrite(size_t  size,
                                     size_t* begin_offset,
                                     size_t* data_offset,
                                     size_t* end_offset) {
  sem_wait(&alloc_lock_);
  auto id = write_block_id_++;
  auto ring_offset = alloc_offset_ % max_buf_size_;
  auto tail_frag_size = max_buf_size_ - ring_offset;
  *begin_offset = alloc_offset_;
  if (tail_frag_size < size) {
    alloc_offset_ += tail_frag_size;
  }
  *data_offset = alloc_offset_;
  alloc_offset_ += size;
  *end_offset = alloc_offset_;
  Check(*end_offset - *begin_offset < max_buf_size_, "message is too large!");
  sem_post(&alloc_lock_);
  return id;
}

size_t ShmQueueMeta::GetBlockToRead() {
  return __sync_fetch_and_add(&read_block_id_, 1);
}

void ShmQueueMeta::ReleaseBlock(size_t id) {
  sem_wait(&release_lock_);
  GetBlockMeta(id).release = true;
  while (id < read_block_id_) {
    auto& block = GetBlockMeta(id);
    if (block.release && block.begin == released_offset_) {
      released_offset_ = block.end;
      block.release = false;
      block.NotifyToWrite();
    } else {
      break;
    }
    id++;
  }
  sem_post(&release_lock_);
}

ShmQueueMeta::BlockMeta& ShmQueueMeta::GetBlockMeta(size_t id) {
  auto* meta = reinterpret_cast<BlockMeta*>(
      reinterpret_cast<char*>(this) + block_meta_offset_);
  return meta[id % max_block_num_];
}

void* ShmQueueMeta::GetData(size_t offset) {
  return reinterpret_cast<char*>(this)
      + data_buf_offset_ + offset % max_buf_size_;
}

ShmQueue::ShmQueue(size_t max_block_num, size_t max_buf_size) {
  max_block_num_ = max_block_num;
  max_buf_size_ = max_buf_size;
  shm_size_ = max_buf_size_ + sizeof(ShmQueueMeta)
      + sizeof(ShmQueueMeta::BlockMeta) * max_block_num_;
  shmid_ = shmget(IPC_PRIVATE, shm_size_, 0666 | IPC_CREAT | IPC_EXCL);
  Check(shmid_ != -1, "shmget failed!");
  void* shmp = shmat(shmid_, (void *)0, 0);
  Check(shmp != (void *)-1, "shmat failed!");
  auto* meta_ptr = reinterpret_cast<ShmQueueMeta*>(shmp);
  meta_ = std::shared_ptr<ShmQueueMeta>(meta_ptr, ShmQueueMetaDeleter{shmid_});
  meta_->Initialize(max_block_num_, max_buf_size_);
}

ShmQueue::ShmQueue(int shmid) {
  Check(shmid != -1, "invalid shmid!");
  shmid_ = shmid;
  void* shmp = shmat(shmid_, (void *)0, 0);
  Check(shmp != (void *)-1, "shmat failed!");
  auto* meta_ptr = reinterpret_cast<ShmQueueMeta*>(shmp);
  meta_ = std::shared_ptr<ShmQueueMeta>(meta_ptr, ShmQueueMetaDeleter{-1});
  max_block_num_ = meta_->max_block_num_;
  max_buf_size_ = meta_->max_buf_size_;
  shm_size_ = max_buf_size_ + sizeof(ShmQueueMeta)
      + sizeof(ShmQueueMeta::BlockMeta) * max_block_num_;
}

ShmQueue::ShmQueue(ShmQueue&& other) noexcept
    : max_block_num_(other.max_block_num_),
      max_buf_size_(other.max_buf_size_),
      shm_size_(other.shm_size_),
      shmid_(other.shmid_),
      meta_(std::move(other.meta_)) {
  other.max_block_num_ = 0;
  other.max_buf_size_ = 0;
  other.shm_size_ = 0;
  other.shmid_ = -1;
}

ShmQueue& ShmQueue::operator=(ShmQueue&& other) noexcept {
  if (this != &other) {
    max_block_num_ = other.max_block_num_;
    max_buf_size_ = other.max_buf_size_;
    shm_size_ = other.shm_size_;
    shmid_ = other.shmid_;
    meta_ = std::move(other.meta_);
    other.max_block_num_ = 0;
    other.max_buf_size_ = 0;
    other.shm_size_ = 0;
    other.shmid_ = -1;
  }
  return *this;
}

void ShmQueue::Enqueue(const void* data, size_t size) {
  Enqueue(size, [data, size] (void* shm_write_ptr) {
    std::memcpy(shm_write_ptr, data, size);
  });
}

void ShmQueue::Enqueue(size_t size, WriteFunc func) {
  size_t begin_offset, data_offset, end_offset;
  auto block_id = meta_->GetBlockToWrite(
      size, &begin_offset, &data_offset, &end_offset);
  // Check for ring buffer conflicts.
  while (block_id >= meta_->read_block_id_ + max_block_num_ ||
         end_offset >= meta_->released_offset_ + max_buf_size_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  auto& block = meta_->GetBlockMeta(block_id);
  block.WaitForWriting();

  auto* shm_write_ptr = meta_->GetData(data_offset);
  func(shm_write_ptr);

  block.begin = begin_offset;
  block.data = data_offset;
  block.end = end_offset;

  block.NotifyToRead();
}

ShmData ShmQueue::Dequeue(unsigned int timeout_ms) {
  auto timeout_duration = std::chrono::milliseconds(timeout_ms);
  auto start_time = std::chrono::steady_clock::now();
  while (meta_->read_block_id_ >= meta_->write_block_id_) {
    if (timeout_ms > 0) {
      auto elapsed_time = std::chrono::steady_clock::now() - start_time;
      if (elapsed_time > timeout_duration) {
        throw QueueTimeoutError();
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto block_id = meta_->GetBlockToRead();

  auto& block = meta_->GetBlockMeta(block_id);
  block.WaitForReading();

  auto* shm_read_ptr = meta_->GetData(block.data);
  auto shm_data_size = block.end - block.data;
  return {shm_read_ptr, shm_data_size, block_id, meta_};
}

bool ShmQueue::Empty() {
  return meta_->read_block_id_ == meta_->write_block_id_;
}

void ShmQueue::PinMemory() {
#ifdef WITH_CUDA
  cudaHostRegister(meta_.get(), shm_size_, cudaHostRegisterMapped);
  CUDACheckError();
#endif
}

void ShmQueue::ShmQueueMetaDeleter::operator()(ShmQueueMeta* meta_ptr) {
  if (meta_ptr) {
    if (shmid > 0) {
      meta_ptr->Finalize();
    }
    Check(shmdt(meta_ptr) != -1, "shmdt failed!");
  }
  if (shmid > 0) {
    Check(shmctl(shmid, IPC_RMID, 0) != -1, "shmctl(IPC_RMID) failed!");
  }
}

}  // namespace graphlearn_torch
