
#ifndef GRAPHLEARN_TORCH_CSRC_RPC_DIST_SERVER_GRPC_H_
#define GRAPHLEARN_TORCH_CSRC_RPC_DIST_SERVER_GRPC_H_

#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <thread>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <torch/extension.h>

#include "graphlearn_torch/csrc/rpc/server_threadpool.h"
#include "graphlearn_torch/csrc/rpc/generated/helloworld.grpc.pb.h"

namespace graphlearn_torch{
class ServerImpl final {
  public:
    ~ServerImpl() {
      server_->Shutdown();
      // Always shutdown the completion queue after the server.
      cq_->Shutdown();
      pool->~ThreadPool();
    }
 
    // There is no shutdown handling in this code.
    void Run(std::string addr, int thread_num);
 
 private:
    void HandleRpcs();
 
    std::unique_ptr<grpc::ServerCompletionQueue> cq_;
    glt::Greeter::AsyncService service_;
    std::unique_ptr<grpc::Server> server_;

    std::shared_ptr<ThreadPool> pool;
    std::string addr;
};
 
class CallDataBase{
  public:
    virtual void Proceed() = 0;
};

class CallData : public CallDataBase{
  public:
    // Take in the "service" instance (in this case representing an asynchronous
    // server) and the completion queue "cq" used for asynchronous communication
    // with the gRPC runtime.
    CallData(glt::Greeter::AsyncService* service, grpc::ServerCompletionQueue* cq)
        : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {}
    virtual void Proceed() = 0;
protected:
// The means of communication with the gRPC runtime for an asynchronous
    // service.
    glt::Greeter::AsyncService* service_;
    // The producer-consumer queue where for asynchronous server notifications.
    grpc::ServerCompletionQueue* cq_;
    // Context for the rpc, allowing to tweak aspects of it such as the use
    // of compression, authentication, as well as to send metadata back to the
    // client.
    grpc::ServerContext ctx_;
 
    // What we get from the client.
    glt::HelloRequest request_;
    // What we send back to the client.
    glt::HelloReply reply_;
 
    // The means to get back to the client.
    grpc::ServerAsyncResponseWriter<glt::HelloReply> responder_;
 
    // Implement a tiny state machine with the following states.
    enum CallStatus { CREATE, PROCESS, FINISH };
    CallStatus status_;  // The current serving state.

    std::string addr;
};
 
class HelloCall : public CallData {
  public:
    void Proceed();
    HelloCall(glt::Greeter::AsyncService* service, grpc::ServerCompletionQueue* cq, std::string addr)
        : CallData (service, cq){
          this->addr = addr;
          Proceed();
    }
};
} // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_CSRC_RPC_DIST_SERVER_GRPC_H_