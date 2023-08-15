
# from .. import py_graphlearn_torch as pywrap

# class DistSamplingGrpcClient(object):
#     def __init__(self,
#                  server_ip1: str,
#                  server_port1: int,
#                  server_ip2: str,
#                  server_port2: int,
#                  max_send_message_size: int = 100 * 1024 * 1024,
#                  max_recv_message_size: int = 100 * 1024 * 1024
#                 ) -> None:
#         self.client = pywrap.GreeterClient(server_ip1 + ":" + str(server_port1), server_ip2 + ":" + str(server_port2), max_send_message_size)
#         # res = client.SayHello("world")
#         # print(res)
#     def SayHello(self, name):
#         self.client.say_hello_again(name)

#     def request_shutdown(self):
#         print("Client: Shutting down server")
        

#     def request_wait_for_exit(self):
#         print("Client: Waiting for exit")

#     def request_exit(self):
#         print("Client: Exiting")

#     def request_get_dataset_meta(self):
#         print("Client: Getting dataset meta")
#         response = self.stub.get_dataset_meta(self.empty_msg)

#     def request_create_sampling_producer(self):
#         print("Client: Creating sampling producer")
        
#     def request_destroy_sampling_producer(self):
#         print("Client: Destroying sampling producer")
        
#     def request_start_new_epoch_sampling(self):
#         print("Client: Starting new epoch sampling")

#     def request_fetch_one_sampled_message(self, size):
#         print("[Client] Fetching one sampled message")
#         self.client.fetch_data(size)
#         # time1 = time.time()
#         # response = self.stub.fetch_one_sampled_message(dist_sampling_pb2.IntMessage(value=10_000_000))
#         # time2 = time.time()
#         # tensor = torch.from_numpy(np.frombuffer(b''.join(response.values), dtype=np.int32))
#         # time3 = time.time()
#         # print(f'生成tensor时间:{(time3-time2)*1000}, 总时间：{int((time2-time1)*1000)}ms')
#         # print(f'数据生成时间:{response.dtype}ms, 数据tolist时间:{response.length}ms, 总时间(不包括数据生成时间):{int((time2-time1)*1000)-int(response.dtype)}')
    
        