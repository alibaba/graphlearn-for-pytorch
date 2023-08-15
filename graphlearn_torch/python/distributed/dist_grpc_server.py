
from .. import py_graphlearn_torch as pywrap

class DistSamplingServer():
  """ Provides methods that implement functionality of distributed sampling server.
  """
  def __init__(self, addr: str, port: int, num_thread: int) -> None:
    self.addr = addr
    self.port = port
    self.num_thread = num_thread 
    
  def start(self):
    self.server = pywrap.ServerImpl()
    self.server.run(self.addr + ":" + str(self.port), self.num_thread)

  def shutdown(self, request, unused_context):
    print("Server: Shutting down server")

  def wait_for_exit(self, request, unused_context):
    print("Server: Waiting for exit")
  
  def exit(self, request, unused_context):
    print("Server: Exiting")

  def get_dataset_meta(self, request, unused_context):
    print("Server: Getting dataset meta")

  def create_sampling_producer(self, request, unused_context):
    print("Server: Creating sampling producer")

  def destroy_sampling_producer(self, request, unused_context):
    print("Server: Destroying sampling producer")

  def start_new_epoch_sampling(self, request, unused_context):
    print("Server: Starting new epoch sampling")
 
  def fetch_one_sampled_message(self, request, unused_context):
    print("Server: Fetching one sampled message")
    