#include "graphlearn_torch/csrc/rpc/dist_server_grpc.h"

namespace graphlearn_torch {

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ResourceQuota;
using grpc::ServerContext;
using grpc::Status;
using glt::Greeter;
using glt::HelloReply;
using glt::HelloRequest;

void ServerImpl::Run(std::string addr, int thread_num) {
  std::string server_address(addr);
  this->addr = addr;
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service_);
  this->cq_ = builder.AddCompletionQueue();
  this->server_ = builder.BuildAndStart();
  this->pool = std::make_shared<ThreadPool>(thread_num);
  
  // std::cout << "Server listening on " << server_address << std::endl;
  HandleRpcs();
}
 
// This can be run in multiple threads if needed.
void ServerImpl::HandleRpcs() {
  // Spawn a new CallData instance to serve new clients.
  new HelloCall(&service_, cq_.get(), this->addr);
  void* tag;  // uniquely identifies a request.
  bool ok;
  while (true) {
    // Block waiting to read the next event from the completion queue. The
    // event is uniquely identified by its tag, which in this case is the
    // memory address of a CallData instance.
    // The return value of Next should always be checked. This return value
    // tells us whether there is any kind of event or cq_ is shutting down.
    GPR_ASSERT(cq_->Next(&tag, &ok));
    GPR_ASSERT(ok);
    // start a new thread to proceed
    // static int i = 1;
    // std::cout<<"[Server] start a new thread to proceed-"<<i++<<", server addr:"<<addr<<std::endl;
    this->pool->enqueue([tag] {
      static_cast<CallDataBase*>(tag)->Proceed();
    });
  }
}

void HelloCall::Proceed() {
  if (status_ == CREATE) {
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;
 
    // As part of the initial CREATE state, we *request* that the system
    // start processing SayHello requests. In this request, "this" acts are
    // the tag uniquely identifying the request (so that different CallData
    // instances can serve different requests concurrently), in this case
    // the memory address of this CallData instance.
    service_->RequestSayHelloAgain(&ctx_, &request_, &responder_, cq_, cq_, this);

  } else if (status_ == PROCESS) {
    // Spawn a new CallData instance to serve new clients while we process
    // the one for this CallData. The instance will deallocate itself as
    // part of its FINISH state.
    new HelloCall(service_, cq_, this->addr);
 
    // The actual processing.
    std::string prefix("Hello ");
    std::this_thread::sleep_for(std::chrono::seconds(2));
    reply_.set_message(prefix + request_.name() + this->addr);
 
    // And we are done! Let the gRPC runtime know we've finished, using the
    // memory address of this instance as the uniquely identifying tag for
    // the event.
    status_ = FINISH;
    responder_.Finish(reply_, Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    // Once in the FINISH state, deallocate ourselves (CallData).
    delete this;
  }
}
}  // namespace graphlearn_torch