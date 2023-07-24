/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <memory>
#include <string>
#include <chrono>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>

#include "graphlearn_torch/csrc/rpc/generated/helloworld.grpc.pb.h"



using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using glt::Greeter;
using glt::HelloReply;
using glt::HelloRequest;

class GreeterClient {
 public:
  GreeterClient(std::shared_ptr<Channel> channel)
      : stub_(Greeter::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  std::string SayHello(const std::string& user) {
    // Data we are sending to the server.
    HelloRequest request;
    request.set_name(user);

    // Container for the data we expect from the server.
    HelloReply reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->SayHello(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      return reply.message();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC failed";
    }
  }
  std::string SayHelloAgain(const std::string& user) {
    // Follows the same pattern as SayHello.
    HelloRequest request;
    request.set_name(user);
    HelloReply reply;
    ClientContext context;

    // Here we can use the stub's newly available method we just added.
    Status status = stub_->SayHelloAgain(&context, request, &reply);
    if (status.ok()) {
      return reply.message();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC failed";
    }
  }
  int FetchData(int size) {
    // Follows the same pattern as SayHello.
    ::glt::Intmsg msg;
    msg.set_data(size);
    ::glt::Bytesmsg* reply = new ::glt::Bytesmsg();
    ClientContext context;

    // Here we can use the stub's newly available method we just added.
    Status status = stub_->FetchData(&context, msg, reply);
    std::string binary_str;
    std::cout << reply->ByteSizeLong() << std::endl;
    if (status.ok()) {
      return reply->data_size();
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return -1;
    }
  }

 private:
  std::unique_ptr<Greeter::Stub> stub_;
};

// int main(int argc, char** argv) {
//   int max_message_size = 100 * 1024 * 1024; // 100 MB
//   grpc::ChannelArguments args;
//   args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, max_message_size);
//   args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, max_message_size);

//   // Instantiate the client. It requires a channel, out of which the actual RPCs
//   // are created. This channel models a connection to an endpoint specified by
//   // the argument "--target=" which is the only expected argument.
//   std::string target_str = "localhost:50051";
//   // We indicate that the channel isn't authenticated (use of
//   // InsecureChannelCredentials()).
//   // GreeterClient greeter(
//   //     grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
//   GreeterClient greeter(
//       grpc::CreateCustomChannel(target_str, grpc::InsecureChannelCredentials(), args));
  
//   std::string user("world");
//   std::string reply = greeter.SayHello(user);
//   std::cout << "Greeter received: " << reply << std::endl;

//   reply = greeter.SayHelloAgain(user);
//   std::cout << "Greeter received: " << reply << std::endl;

//   std::chrono::steady_clock::time_point start_time, end_time;

//   start_time = std::chrono::steady_clock::now();
//   greeter.FetchData(100000000);
//   end_time = std::chrono::steady_clock::now();

//   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
//   std::cout << "Total Time taken: " << duration.count() << " ms" << std::endl;

//   return 0;
// }
