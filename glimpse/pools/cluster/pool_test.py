# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.pools.cluster.pool import ClusterPool
from glimpse.pools.cluster.manager import ClusterManager
from glimpse.util.zmq_cluster import Connect, Worker
import threading
import time
import unittest
import zmq

def WorkerThreadTarget(context, request_receiver, result_sender,
    command_receiver):

  def worker_callback(dynamic_batch_request):
    function, batch_request = dynamic_batch_request
    return map(function, batch_request)

  worker = Worker(context, request_receiver, result_sender, worker_callback,
      command_receiver)
  worker.Setup()
  worker.Run()

def PicklableFunction(x):
  return x * 100

class TestClusterPool(unittest.TestCase):

  def test_imap_unordered(self):

    context = zmq.Context()
    # bind the command_sender socket
    command_sender = context.socket(zmq.PUB)
    command_sender.bind("inproc://commands")
    # launch the cluster manager
    #   request_sender and result_receiver bind
    #   command_receiver connects
    request_sender = Connect("inproc://requests", bind = True)
    result_receiver = Connect("inproc://results", bind = True)
    command_receiver = Connect("inproc://commands",
        options = {zmq.SUBSCRIBE : ""})  # subscribe to all command messages
    manager = ClusterManager(context, request_sender, result_receiver,
        command_sender, command_receiver, use_threading = True)
    manager.Setup()
    time.sleep(1)
    # launch the worker
    #   all sockets connect
    request_receiver = Connect("inproc://requests")
    result_sender = Connect("inproc://results")
    worker = threading.Thread(target = WorkerThreadTarget, args = (context,
        request_receiver, result_sender, command_receiver))
    worker.daemon = True
    worker.start()
    xs = range(10)
    pool = ClusterPool(manager = manager)
    expected_ys = map(PicklableFunction, xs)
    actual_ys = sorted(pool.imap_unordered(PicklableFunction, xs))
    self.assertEqual(expected_ys, actual_ys)
    # tear down the worker
    Worker.SendKillCommand(context, command_sender)
    worker.join(100)
    self.assertFalse(worker.is_alive())
    # tear down the cluster manager
    manager.Shutdown()
    self.assertFalse(manager.sink.is_alive())

if __name__ == '__main__':

  # Uncomment the following for debugging messages.
  #logging.getLogger().setLevel(logging.INFO)

  unittest.main()
