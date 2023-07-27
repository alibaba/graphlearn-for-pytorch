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

#include <string>
#include <unordered_set>

#include "gtest/gtest.h"

#include "graphlearn_torch/include/shm_queue.h"

using namespace graphlearn_torch;

struct TestMsg {
  int id;
  std::string content;

  TestMsg(int id, std::string content): id(id), content(std::move(content)) {}

  std::string Dump() {
    char id_str[sizeof(int) + content.size()];
    memcpy(id_str, &id, sizeof(int));
    memcpy(id_str + sizeof(int), content.data(), content.size());
    return {id_str, sizeof(int) + content.size()};
  }

  static TestMsg Load(const ShmData& shm_data) {
    auto id = *static_cast<const int*>(shm_data.Data());
    auto* content_ptr = static_cast<const char*>(shm_data.Data()) + sizeof(int);
    auto content_len = shm_data.Length() - sizeof(int);
    return {id, {content_ptr, content_len}};
  }
};

struct VerifyMsg {
  enum BehvType : int { Send, Receive };

  int id;
  BehvType type;

  VerifyMsg(int id, BehvType type) : id(id), type(type) {}
};

class ShmQueueTest : public ::testing::Test {
protected:
  void SetUp() override {
    shq_ = new ShmQueue{5, 256};
    res_shq_ = new ShmQueue{40, 1024};
  }

  void TearDown() override {
    delete shq_;
    delete res_shq_;
  }

protected:
  ShmQueue* shq_ = nullptr;
  ShmQueue* res_shq_ = nullptr;
};

TEST_F(ShmQueueTest, Functionality) {
  pid_t child_pid;
  for (int i = 0; i < 4; i++) {
    child_pid = fork();
    if (child_pid == 0 || child_pid == -1) break;
  }

  EXPECT_NE(child_pid, -1);

  if (child_pid == 0) {
    // In child process
    if (getpid() % 2 == 0) {
      // Sender
      std::string value = "base";
      for (int r = 0; r < 10; r++) {
        // append new value
        value += ":" + std::to_string(r);
        int msg_id = getpid() * 100 + r;
        std::string content = "this is a shm message, value: " + value;
        // send verify msg
        VerifyMsg verify_msg{msg_id, VerifyMsg::BehvType::Send};
        res_shq_->Enqueue(&verify_msg, sizeof(VerifyMsg));
        // send test msg
        TestMsg test_msg{msg_id, content};
        auto serialized_test_msg = test_msg.Dump();
        shq_->Enqueue(serialized_test_msg.data(), serialized_test_msg.size());
      }
    } else {
      // Receiver
      for (int r = 0; r < 10; r++) {
        // receive test msg
        auto shm_data = shq_->Dequeue();
        auto test_msg = TestMsg::Load(shm_data);
        // send verify msg
        VerifyMsg verify_msg{test_msg.id, VerifyMsg::BehvType::Receive};
        res_shq_->Enqueue(&verify_msg, sizeof(VerifyMsg));
      }
    }
    exit(EXIT_SUCCESS);
  } else {
    // In parent process
    // wait all child processes
    int status;
    unsigned wait_cnt = 0;
    while (wait(&status) > 0) {
      wait_cnt++;
    }
    EXPECT_EQ(wait_cnt, 4);

    std::unordered_set<int> sent_msgs, received_msgs;
    for (int i = 0; i < 40; i++) {
      EXPECT_FALSE(res_shq_->Empty());
      auto shm_data = res_shq_->Dequeue();
      auto* verify_msg = static_cast<const VerifyMsg*>(shm_data.Data());
      if (verify_msg->type == VerifyMsg::BehvType::Send) {
        EXPECT_EQ(sent_msgs.count(verify_msg->id), 0);
        sent_msgs.insert(verify_msg->id);
      } else {
        EXPECT_EQ(sent_msgs.count(verify_msg->id), 1);
        EXPECT_EQ(received_msgs.count(verify_msg->id), 0);
        received_msgs.insert(verify_msg->id);
      }
    }
    try {
      auto shm_data = res_shq_->Dequeue(10);
    }
    catch (const QueueTimeoutError& e) {
      std::cout << "Expected QueueTimeoutError: " << e.what() << std::endl;
    }
    EXPECT_TRUE(res_shq_->Empty());
    EXPECT_EQ(sent_msgs.size(), 20);
    EXPECT_EQ(received_msgs.size(), 20);
  }
}
