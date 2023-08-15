import unittest
import multiprocessing
import time
import asyncio
import grpc
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../graphlearn_torch/csrc/rpc/generated"))

import helloworld_pb2
import helloworld_pb2_grpc

import graphlearn_torch as glt

class GraphTestCase(unittest.TestCase):
    def setUp(self):
      
        server = glt.distributed.dist_grpc_server.DistSamplingServer("localhost", 50051, 10)
        # start server in a new process
        self.server_process = multiprocessing.Process(target=server.start)
        self.server_process.start()
        # wait for server to start
        time.sleep(2)
        
    def test_helloworld(self):
        self.resps = []
        self.async_call_times = 30
        asyncio.run(self.async_call())
        
        # check response
        self.resps = [int(resp.split()[2]) for resp in self.resps]
        print(self.resps)
        self.assertEqual(sorted(self.resps), list(range(self.async_call_times)))
        # exit
        self.server_process.terminate()
        
    async def run_once(self, req, stub):
        response = await stub.SayHelloAgain(req)
        self.resps.append(response.message)
            
    async def async_call(self):
        # options = [('grpc.max_send_message_length', 100_000_010), ('grpc.max_receive_message_length', 100_000_010)]
        options = []
        channel =  grpc.aio.insecure_channel('localhost:50051', options=options)
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        
        reqs = [self.run_once(helloworld_pb2.HelloRequest(name=f'glt {i} '), stub) for i in range(self.async_call_times)]
        await asyncio.gather(*reqs) 
        

if __name__ == "__main__":
  unittest.main()