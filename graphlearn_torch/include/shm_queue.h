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

#include <unistd.h>
#include <semaphore.h>

#include <functional>
#include <memory>
#include <stdexcept>

#ifndef GRAPHLEARN_TORCH_INCLUDE_SHM_QUEUE_H_
#define GRAPHLEARN_TORCH_INCLUDE_SHM_QUEUE_H_

namespace graphlearn_torch {

struct ShmQueueMeta;

class ShmData {
public:
  ShmData() : data_(nullptr), len_(0), block_id_(0), meta_() {}
  ShmData(void* data, size_t len, size_t block_id,
          const std::shared_ptr<ShmQueueMeta>& meta)
      : data_(data), len_(len), block_id_(block_id), meta_(meta) {}
  ~ShmData();

  /// Disable copy constructor to make sure exact one block release.
  ShmData(const ShmData&) = delete;
  ShmData& operator=(const ShmData&) = delete;

  /// Move only.
  ShmData(ShmData&& other) noexcept;
  ShmData& operator=(ShmData&& other) noexcept;

  void* Data() {
    return data_;
  }

  const void* Data() const {
    return data_;
  }

  size_t Length() const {
    return len_;
  }

private:
  void*  data_;
  size_t len_;
  size_t block_id_;
  std::shared_ptr<ShmQueueMeta> meta_;
};

class ShmQueueMeta {
  /// We use a byte array to simulate a ring buffer and allocate blocks
  /// with different size, the blocks will be stored in the buffer one by
  /// one.
  ///
  /// When the current allocation pointer is near the array tail and the
  /// remaining size at the tail cannot satisfy the size of an allocation
  /// request, we will skip this "tail fragment" and allocate the block from
  /// the beginning of buffer. However, the "tail fragment" is also part of
  /// this block and should be free when this block has been consumed.
  ///
  struct BlockMeta {
    // The beginning offset of current block. including tail fragment if exists.
    size_t begin;
    // The true offset of block data, skipping tail fragment.
    size_t data;
    // The ending offset of current block.
    size_t end;
    // Semaphore of write operation, should be wait before writing.
    sem_t write;
    // Semaphore of read operation, should be wait before reading.
    sem_t read;
    // Release flag, when releasing this block, if any block in front of it
    // has not been released, we cannot release the buffer immediately as we
    // need to make sure a consecutive buffer.
    //
    // Instead, set this flag to make this block can be released by the
    // previous block afterward.
    //
    bool release;

    void Initialize() {
      release = false;
      sem_init(&write, 1, 1);
      sem_init(&read, 1, 0);
    }

    void Finalize() {
      sem_destroy(&write);
      sem_destroy(&read);
    }

    void WaitForWriting() {
      sem_wait(&write);
    }

    void NotifyToWrite() {
      sem_post(&write);
    }

    void WaitForReading() {
      sem_wait(&read);
    }

    void NotifyToRead() {
      sem_post(&read);
    }
  };

public:
  ShmQueueMeta() = default;

  /// Initialize the shm buffer.
  void Initialize(size_t max_block_num, size_t max_buf_size);

  /// Finalize the shm buffer.
  void Finalize();

  /// Get a block allocation to write with request size.
  /// \return block id
  size_t GetBlockToWrite(size_t  size,
                         size_t* begin_offset,
                         size_t* data_offset,
                         size_t* end_offset);

  /// Get a block to read.
  /// \return block id
  size_t GetBlockToRead();

  /// Release a block with block id.
  void ReleaseBlock(size_t id);

  /// Access block meta with block id.
  BlockMeta& GetBlockMeta(size_t id);

  /// Get data ptr of specific offset in ring buffer
  /// \return ptr
  void* GetData(size_t offset);

private:
  size_t     max_block_num_;
  size_t     max_buf_size_;
  size_t     block_meta_offset_;
  size_t     data_buf_offset_;
  size_t     write_block_id_;
  size_t     read_block_id_;
  size_t     alloc_offset_;
  size_t     released_offset_;
  sem_t      alloc_lock_;
  sem_t      release_lock_;
  friend class ShmQueue;
};

/// Shared-Memory Queue should be constructed and destructed on main process.
class ShmQueue {
public:
  /// Create a new ShmQueue and allocate a new shared memory buffer.
  ///
  /// \param max_block_num The max number of buffered messages in queue.
  /// \param max_buf_size The max size of queue capacity in bytes.
  ///
  ShmQueue(size_t max_block_num, size_t max_buf_size);

  /// Create a ShmQueue instance from an allocated shared memory buffer with
  /// a "shmid" returned by system `shmget` call.
  explicit ShmQueue(int shmid);

  ~ShmQueue() = default;

  /// Disable copy constructor to ensure exact one owner.
  ShmQueue(const ShmQueue&) = delete;
  ShmQueue& operator=(const ShmQueue&) = delete;

  /// Move only.
  ShmQueue(ShmQueue&& other) noexcept;
  ShmQueue& operator=(ShmQueue&& other) noexcept;

  /// Enqueue a message on child process.
  ///
  /// \param data The data pointer to write.
  /// \param size The data size to write.
  ///
  void Enqueue(const void* data, size_t size);

  /// Enqueue a message on child process.
  ///
  /// \param size The data size to write.
  /// \param func The func to write your data into the acquired buffer pointer.
  ///
  using WriteFunc = std::function<void(void*)>;
  void Enqueue(size_t size, WriteFunc func);

  /// Dequeue a message on child process.
  /// \return `ShmData`
  ShmData Dequeue(unsigned int timeout_ms = 0);

  bool Empty();

  /// Pin memory on child processes.
  void PinMemory();

  /// Get unique shm id of the underlying shm buffer.
  int ShmId() const {
    return shmid_;
  }

private:
  size_t max_block_num_;
  size_t max_buf_size_;
  size_t shm_size_;
  int    shmid_;

  /// Deleter to release shared memory when the underlying meta is shared
  /// by multiple shared_ptrs.
  struct ShmQueueMetaDeleter {
    /// If "shmid > 0", which means current instance of `ShmQueue` is the
    /// creator of the underlying shm, thus, `shmctl` should be called after
    /// detaching the shm in this deleter. Otherwise, current instance is not
    /// the creator and only need to detach the shm with `shmdt`.
    int shmid;
    explicit ShmQueueMetaDeleter(int shmid) : shmid(shmid) {}
    void operator()(ShmQueueMeta* meta_ptr);
  };
  std::shared_ptr<ShmQueueMeta> meta_;
};

class QueueTimeoutError : public std::runtime_error {
public:
    QueueTimeoutError() : std::runtime_error("Timeout: Queue is empty.") {}
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_SHM_QUEUE_H_
