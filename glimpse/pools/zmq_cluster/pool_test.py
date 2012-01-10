# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from .pool import ClusterPool, PoolWorker
from .manager import ClusterManager
from glimpse.util.zmq_cluster import Connect, BasicWorker, ClusterResult, \
    ReceiverTimeoutException
from multiprocessing import Process
import threading
import time
import unittest
import zmq

class MockPoolWorker(BasicWorker):

  def __init__(self, *args, **kwargs):
    super(MockPoolWorker, self).__init__(*args, **kwargs)

  def HandleRequest(self, dynamic_batch_request):
    function, batch_request = dynamic_batch_request
    return map(function, batch_request)

def WorkerThreadTarget(context, request_receiver, result_sender,
    command_receiver):

  worker = MockPoolWorker(context, request_receiver, result_sender,
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
        command_sender, command_receiver, use_threading = True,
        receiver_timeout = None)
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
    BasicWorker.SendKillCommand(context, command_sender)
    worker.join(100)
    self.assertFalse(worker.is_alive())
    # tear down the cluster manager
    manager.Shutdown()
    self.assertFalse(manager.sink.is_alive())

class MockConfig(object):
  def __init__(self):
    self.request_receiver = Connect(url = "inproc://requests", type = zmq.PULL)
    self.result_sender = Connect(url = "inproc://results", type = zmq.PUSH)
    self.command_receiver = Connect(url = "inproc://commands", type = zmq.PULL)

def PoolWorkerTarget(context, config):
  try:
    worker = PoolWorker(context, config, receiver_timeout = 10)
    worker.Setup()
    worker.Run()
  except ReceiverTimeoutException:
    pass

class TestPoolWorker(unittest.TestCase):

  def test_worker(self):
    context = zmq.Context()
    config = MockConfig()
    request_sender = config.request_receiver.MakeSocket(context,
        type = zmq.PUSH, bind = True)
    result_receiver = config.result_sender.MakeSocket(context, type = zmq.PULL,
        bind = True)
    command_sender = config.command_receiver.MakeSocket(context,
        type = zmq.PUSH, bind = True)
    worker = threading.Thread(target = PoolWorkerTarget, args = (context,
        config,), name = "PoolWorkerThread")
    worker.start()
    # send a batch request
    xs = range(10)
    expected = map(PicklableFunction, xs)
    request_sender.send_pyobj((PicklableFunction, xs))
    poller = zmq.Poller()
    poller.register(result_receiver, zmq.POLLIN)
    sock = poller.poll(timeout = 10)
    self.assertNotEquals(len(sock), 0)
    result = result_receiver.recv_pyobj()
    self.assertEqual(result.status, ClusterResult.STATUS_SUCCESS)
    actual = result.payload
    self.assertEqual(expected, actual)

if __name__ == '__main__':

  # Uncomment the following for debugging messages.
  #logging.getLogger().setLevel(logging.INFO)

  # Uncomment the following to run a subset of tests.
  #~ suite = unittest.TestLoader().loadTestsFromTestCase(TestPoolWorker)
  #~ unittest.TextTestRunner(verbosity=2).run(suite)

  unittest.main()
