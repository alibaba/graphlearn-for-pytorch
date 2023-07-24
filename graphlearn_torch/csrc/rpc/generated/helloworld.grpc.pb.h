// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: helloworld.proto
#ifndef GRPC_helloworld_2eproto__INCLUDED
#define GRPC_helloworld_2eproto__INCLUDED

#include "helloworld.pb.h"

#include <functional>
#include <grpcpp/generic/async_generic_service.h>
#include <grpcpp/support/async_stream.h>
#include <grpcpp/support/async_unary_call.h>
#include <grpcpp/support/client_callback.h>
#include <grpcpp/client_context.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/support/message_allocator.h>
#include <grpcpp/support/method_handler.h>
#include <grpcpp/impl/proto_utils.h>
#include <grpcpp/impl/rpc_method.h>
#include <grpcpp/support/server_callback.h>
#include <grpcpp/impl/server_callback_handlers.h>
#include <grpcpp/server_context.h>
#include <grpcpp/impl/service_type.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/stub_options.h>
#include <grpcpp/support/sync_stream.h>

namespace glt {

// The greeting service definition.
class Greeter final {
 public:
  static constexpr char const* service_full_name() {
    return "glt.Greeter";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    // Sends a greeting
    virtual ::grpc::Status SayHello(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::glt::HelloReply* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>> AsyncSayHello(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>>(AsyncSayHelloRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>> PrepareAsyncSayHello(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>>(PrepareAsyncSayHelloRaw(context, request, cq));
    }
    virtual ::grpc::Status SayHelloAgain(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::glt::HelloReply* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>> AsyncSayHelloAgain(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>>(AsyncSayHelloAgainRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>> PrepareAsyncSayHelloAgain(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>>(PrepareAsyncSayHelloAgainRaw(context, request, cq));
    }
    virtual ::grpc::Status FetchData(::grpc::ClientContext* context, const ::glt::Intmsg& request, ::glt::Bytesmsg* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::Bytesmsg>> AsyncFetchData(::grpc::ClientContext* context, const ::glt::Intmsg& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::Bytesmsg>>(AsyncFetchDataRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::Bytesmsg>> PrepareAsyncFetchData(::grpc::ClientContext* context, const ::glt::Intmsg& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::glt::Bytesmsg>>(PrepareAsyncFetchDataRaw(context, request, cq));
    }
    class async_interface {
     public:
      virtual ~async_interface() {}
      // Sends a greeting
      virtual void SayHello(::grpc::ClientContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void SayHello(::grpc::ClientContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response, ::grpc::ClientUnaryReactor* reactor) = 0;
      virtual void SayHelloAgain(::grpc::ClientContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response, std::function<void(::grpc::Status)>) = 0;
      virtual void SayHelloAgain(::grpc::ClientContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response, ::grpc::ClientUnaryReactor* reactor) = 0;
      virtual void FetchData(::grpc::ClientContext* context, const ::glt::Intmsg* request, ::glt::Bytesmsg* response, std::function<void(::grpc::Status)>) = 0;
      virtual void FetchData(::grpc::ClientContext* context, const ::glt::Intmsg* request, ::glt::Bytesmsg* response, ::grpc::ClientUnaryReactor* reactor) = 0;
    };
    typedef class async_interface experimental_async_interface;
    virtual class async_interface* async() { return nullptr; }
    class async_interface* experimental_async() { return async(); }
   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>* AsyncSayHelloRaw(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>* PrepareAsyncSayHelloRaw(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>* AsyncSayHelloAgainRaw(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::glt::HelloReply>* PrepareAsyncSayHelloAgainRaw(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::glt::Bytesmsg>* AsyncFetchDataRaw(::grpc::ClientContext* context, const ::glt::Intmsg& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::glt::Bytesmsg>* PrepareAsyncFetchDataRaw(::grpc::ClientContext* context, const ::glt::Intmsg& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());
    ::grpc::Status SayHello(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::glt::HelloReply* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>> AsyncSayHello(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>>(AsyncSayHelloRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>> PrepareAsyncSayHello(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>>(PrepareAsyncSayHelloRaw(context, request, cq));
    }
    ::grpc::Status SayHelloAgain(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::glt::HelloReply* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>> AsyncSayHelloAgain(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>>(AsyncSayHelloAgainRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>> PrepareAsyncSayHelloAgain(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>>(PrepareAsyncSayHelloAgainRaw(context, request, cq));
    }
    ::grpc::Status FetchData(::grpc::ClientContext* context, const ::glt::Intmsg& request, ::glt::Bytesmsg* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::Bytesmsg>> AsyncFetchData(::grpc::ClientContext* context, const ::glt::Intmsg& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::Bytesmsg>>(AsyncFetchDataRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::Bytesmsg>> PrepareAsyncFetchData(::grpc::ClientContext* context, const ::glt::Intmsg& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::glt::Bytesmsg>>(PrepareAsyncFetchDataRaw(context, request, cq));
    }
    class async final :
      public StubInterface::async_interface {
     public:
      void SayHello(::grpc::ClientContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response, std::function<void(::grpc::Status)>) override;
      void SayHello(::grpc::ClientContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response, ::grpc::ClientUnaryReactor* reactor) override;
      void SayHelloAgain(::grpc::ClientContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response, std::function<void(::grpc::Status)>) override;
      void SayHelloAgain(::grpc::ClientContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response, ::grpc::ClientUnaryReactor* reactor) override;
      void FetchData(::grpc::ClientContext* context, const ::glt::Intmsg* request, ::glt::Bytesmsg* response, std::function<void(::grpc::Status)>) override;
      void FetchData(::grpc::ClientContext* context, const ::glt::Intmsg* request, ::glt::Bytesmsg* response, ::grpc::ClientUnaryReactor* reactor) override;
     private:
      friend class Stub;
      explicit async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class async* async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>* AsyncSayHelloRaw(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>* PrepareAsyncSayHelloRaw(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>* AsyncSayHelloAgainRaw(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::glt::HelloReply>* PrepareAsyncSayHelloAgainRaw(::grpc::ClientContext* context, const ::glt::HelloRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::glt::Bytesmsg>* AsyncFetchDataRaw(::grpc::ClientContext* context, const ::glt::Intmsg& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::glt::Bytesmsg>* PrepareAsyncFetchDataRaw(::grpc::ClientContext* context, const ::glt::Intmsg& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_SayHello_;
    const ::grpc::internal::RpcMethod rpcmethod_SayHelloAgain_;
    const ::grpc::internal::RpcMethod rpcmethod_FetchData_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    // Sends a greeting
    virtual ::grpc::Status SayHello(::grpc::ServerContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response);
    virtual ::grpc::Status SayHelloAgain(::grpc::ServerContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response);
    virtual ::grpc::Status FetchData(::grpc::ServerContext* context, const ::glt::Intmsg* request, ::glt::Bytesmsg* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_SayHello : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_SayHello() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_SayHello() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHello(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestSayHello(::grpc::ServerContext* context, ::glt::HelloRequest* request, ::grpc::ServerAsyncResponseWriter< ::glt::HelloReply>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_SayHelloAgain : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_SayHelloAgain() {
      ::grpc::Service::MarkMethodAsync(1);
    }
    ~WithAsyncMethod_SayHelloAgain() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHelloAgain(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestSayHelloAgain(::grpc::ServerContext* context, ::glt::HelloRequest* request, ::grpc::ServerAsyncResponseWriter< ::glt::HelloReply>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_FetchData : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_FetchData() {
      ::grpc::Service::MarkMethodAsync(2);
    }
    ~WithAsyncMethod_FetchData() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status FetchData(::grpc::ServerContext* /*context*/, const ::glt::Intmsg* /*request*/, ::glt::Bytesmsg* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestFetchData(::grpc::ServerContext* context, ::glt::Intmsg* request, ::grpc::ServerAsyncResponseWriter< ::glt::Bytesmsg>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_SayHello<WithAsyncMethod_SayHelloAgain<WithAsyncMethod_FetchData<Service > > > AsyncService;
  template <class BaseClass>
  class WithCallbackMethod_SayHello : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithCallbackMethod_SayHello() {
      ::grpc::Service::MarkMethodCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::glt::HelloRequest, ::glt::HelloReply>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response) { return this->SayHello(context, request, response); }));}
    void SetMessageAllocatorFor_SayHello(
        ::grpc::MessageAllocator< ::glt::HelloRequest, ::glt::HelloReply>* allocator) {
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(0);
      static_cast<::grpc::internal::CallbackUnaryHandler< ::glt::HelloRequest, ::glt::HelloReply>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~WithCallbackMethod_SayHello() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHello(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* SayHello(
      ::grpc::CallbackServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithCallbackMethod_SayHelloAgain : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithCallbackMethod_SayHelloAgain() {
      ::grpc::Service::MarkMethodCallback(1,
          new ::grpc::internal::CallbackUnaryHandler< ::glt::HelloRequest, ::glt::HelloReply>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::glt::HelloRequest* request, ::glt::HelloReply* response) { return this->SayHelloAgain(context, request, response); }));}
    void SetMessageAllocatorFor_SayHelloAgain(
        ::grpc::MessageAllocator< ::glt::HelloRequest, ::glt::HelloReply>* allocator) {
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(1);
      static_cast<::grpc::internal::CallbackUnaryHandler< ::glt::HelloRequest, ::glt::HelloReply>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~WithCallbackMethod_SayHelloAgain() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHelloAgain(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* SayHelloAgain(
      ::grpc::CallbackServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithCallbackMethod_FetchData : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithCallbackMethod_FetchData() {
      ::grpc::Service::MarkMethodCallback(2,
          new ::grpc::internal::CallbackUnaryHandler< ::glt::Intmsg, ::glt::Bytesmsg>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::glt::Intmsg* request, ::glt::Bytesmsg* response) { return this->FetchData(context, request, response); }));}
    void SetMessageAllocatorFor_FetchData(
        ::grpc::MessageAllocator< ::glt::Intmsg, ::glt::Bytesmsg>* allocator) {
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(2);
      static_cast<::grpc::internal::CallbackUnaryHandler< ::glt::Intmsg, ::glt::Bytesmsg>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~WithCallbackMethod_FetchData() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status FetchData(::grpc::ServerContext* /*context*/, const ::glt::Intmsg* /*request*/, ::glt::Bytesmsg* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* FetchData(
      ::grpc::CallbackServerContext* /*context*/, const ::glt::Intmsg* /*request*/, ::glt::Bytesmsg* /*response*/)  { return nullptr; }
  };
  typedef WithCallbackMethod_SayHello<WithCallbackMethod_SayHelloAgain<WithCallbackMethod_FetchData<Service > > > CallbackService;
  typedef CallbackService ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_SayHello : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_SayHello() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_SayHello() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHello(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_SayHelloAgain : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_SayHelloAgain() {
      ::grpc::Service::MarkMethodGeneric(1);
    }
    ~WithGenericMethod_SayHelloAgain() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHelloAgain(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_FetchData : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_FetchData() {
      ::grpc::Service::MarkMethodGeneric(2);
    }
    ~WithGenericMethod_FetchData() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status FetchData(::grpc::ServerContext* /*context*/, const ::glt::Intmsg* /*request*/, ::glt::Bytesmsg* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_SayHello : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_SayHello() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_SayHello() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHello(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestSayHello(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawMethod_SayHelloAgain : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_SayHelloAgain() {
      ::grpc::Service::MarkMethodRaw(1);
    }
    ~WithRawMethod_SayHelloAgain() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHelloAgain(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestSayHelloAgain(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawMethod_FetchData : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_FetchData() {
      ::grpc::Service::MarkMethodRaw(2);
    }
    ~WithRawMethod_FetchData() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status FetchData(::grpc::ServerContext* /*context*/, const ::glt::Intmsg* /*request*/, ::glt::Bytesmsg* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestFetchData(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(2, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawCallbackMethod_SayHello : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawCallbackMethod_SayHello() {
      ::grpc::Service::MarkMethodRawCallback(0,
          new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->SayHello(context, request, response); }));
    }
    ~WithRawCallbackMethod_SayHello() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHello(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* SayHello(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithRawCallbackMethod_SayHelloAgain : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawCallbackMethod_SayHelloAgain() {
      ::grpc::Service::MarkMethodRawCallback(1,
          new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->SayHelloAgain(context, request, response); }));
    }
    ~WithRawCallbackMethod_SayHelloAgain() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status SayHelloAgain(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* SayHelloAgain(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithRawCallbackMethod_FetchData : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawCallbackMethod_FetchData() {
      ::grpc::Service::MarkMethodRawCallback(2,
          new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
                   ::grpc::CallbackServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->FetchData(context, request, response); }));
    }
    ~WithRawCallbackMethod_FetchData() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status FetchData(::grpc::ServerContext* /*context*/, const ::glt::Intmsg* /*request*/, ::glt::Bytesmsg* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* FetchData(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)  { return nullptr; }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_SayHello : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_SayHello() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler<
          ::glt::HelloRequest, ::glt::HelloReply>(
            [this](::grpc::ServerContext* context,
                   ::grpc::ServerUnaryStreamer<
                     ::glt::HelloRequest, ::glt::HelloReply>* streamer) {
                       return this->StreamedSayHello(context,
                         streamer);
                  }));
    }
    ~WithStreamedUnaryMethod_SayHello() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status SayHello(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedSayHello(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::glt::HelloRequest,::glt::HelloReply>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_SayHelloAgain : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_SayHelloAgain() {
      ::grpc::Service::MarkMethodStreamed(1,
        new ::grpc::internal::StreamedUnaryHandler<
          ::glt::HelloRequest, ::glt::HelloReply>(
            [this](::grpc::ServerContext* context,
                   ::grpc::ServerUnaryStreamer<
                     ::glt::HelloRequest, ::glt::HelloReply>* streamer) {
                       return this->StreamedSayHelloAgain(context,
                         streamer);
                  }));
    }
    ~WithStreamedUnaryMethod_SayHelloAgain() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status SayHelloAgain(::grpc::ServerContext* /*context*/, const ::glt::HelloRequest* /*request*/, ::glt::HelloReply* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedSayHelloAgain(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::glt::HelloRequest,::glt::HelloReply>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_FetchData : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_FetchData() {
      ::grpc::Service::MarkMethodStreamed(2,
        new ::grpc::internal::StreamedUnaryHandler<
          ::glt::Intmsg, ::glt::Bytesmsg>(
            [this](::grpc::ServerContext* context,
                   ::grpc::ServerUnaryStreamer<
                     ::glt::Intmsg, ::glt::Bytesmsg>* streamer) {
                       return this->StreamedFetchData(context,
                         streamer);
                  }));
    }
    ~WithStreamedUnaryMethod_FetchData() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status FetchData(::grpc::ServerContext* /*context*/, const ::glt::Intmsg* /*request*/, ::glt::Bytesmsg* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedFetchData(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::glt::Intmsg,::glt::Bytesmsg>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_SayHello<WithStreamedUnaryMethod_SayHelloAgain<WithStreamedUnaryMethod_FetchData<Service > > > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_SayHello<WithStreamedUnaryMethod_SayHelloAgain<WithStreamedUnaryMethod_FetchData<Service > > > StreamedService;
};

}  // namespace glt


#endif  // GRPC_helloworld_2eproto__INCLUDED
