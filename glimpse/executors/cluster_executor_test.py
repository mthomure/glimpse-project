
from glimpse.executors.cluster_executor import ClusterExecutor
from glimpse.executors.executor import ExecutorMap
from glimpse.util.zmq_cluster import Connect, Worker
import logging
import threading
import time
import unittest
import zmq

def WorkerTarget(context, request_receiver, result_sender, callback,
    command_receiver):
  worker = Worker(context, request_receiver, result_sender, callback,
      command_receiver)
  worker.Setup()
  worker.Run()

class TestClusterExecutor(unittest.TestCase):

  def test_quitsOnCommand(self):
    callback = lambda x: x * 100

    context = zmq.Context()
    # bind the command_sender socket
    logging.info("Binding PUB socket to inproc://commands with context %s" % \
        hash(context))
    command_sender = context.socket(zmq.PUB)
    command_sender.bind("inproc://commands")

    # launch the cluster executor
    #   request_sender and result_receiver bind
    #   command_receiver connects
    request_sender = Connect("inproc://requests", bind = True)
    result_receiver = Connect("inproc://results", bind = True)
    command_receiver = Connect("inproc://commands",
        options = {zmq.SUBSCRIBE : ""})  # subscribe to all command messages
    executor = ClusterExecutor(context, request_sender, result_receiver,
        command_sender, command_receiver, use_threading = True)
    executor.Setup()
    time.sleep(1)

    # launch the worker
    #   all sockets connect
    request_receiver = Connect("inproc://requests")
    result_sender = Connect("inproc://results")
    worker = threading.Thread(target = WorkerTarget, args = (context,
        request_receiver, result_sender, callback, command_receiver))
    worker.daemon = True
    worker.start()

    #~ executor.Put(2)
    #~ self.assertEqual(200, executor.Get())

    # tear down the worker
    Worker.SendKillCommand(context, command_sender)
    worker.join(100)
    self.assertFalse(worker.is_alive())

    # tear down the cluster executor
    executor.Shutdown()
    self.assertFalse(executor.sink.is_alive())

  # test that basic Put()/Get() call pair works
  def test_putGet(self):
    callback = lambda x: x * 100

    context = zmq.Context()
    # bind the command_sender socket
    logging.info("Binding PUB socket to inproc://commands with context %s" % \
        hash(context))
    command_sender = context.socket(zmq.PUB)
    command_sender.bind("inproc://commands")

    # launch the cluster executor
    #   request_sender and result_receiver bind
    #   command_receiver connects
    request_sender = Connect("inproc://requests", bind = True)
    result_receiver = Connect("inproc://results", bind = True)
    command_receiver = Connect("inproc://commands",
        options = {zmq.SUBSCRIBE : ""})  # subscribe to all command messages
    executor = ClusterExecutor(context, request_sender, result_receiver,
        command_sender, command_receiver, use_threading = True)
    executor.Setup()
    time.sleep(1)

    # launch the worker
    #   all sockets connect
    request_receiver = Connect("inproc://requests")
    result_sender = Connect("inproc://results")
    worker = threading.Thread(target = WorkerTarget, args = (context,
        request_receiver, result_sender, callback, command_receiver))
    worker.daemon = True
    worker.start()

    executor.Put(2)
    self.assertEqual(200, executor.Get())

    # tear down the worker
    Worker.SendKillCommand(context, command_sender)
    worker.join(100)
    self.assertFalse(worker.is_alive())

    # tear down the cluster executor
    executor.Shutdown()
    self.assertFalse(executor.sink.is_alive())

  # test that map works
  def test_map(self):
    callback = lambda x: x * 100

    context = zmq.Context()
    # bind the command_sender socket
    logging.info("Binding PUB socket to inproc://commands with context %s" % \
        hash(context))
    command_sender = context.socket(zmq.PUB)
    command_sender.bind("inproc://commands")

    # launch the cluster executor
    #   request_sender and result_receiver bind
    #   command_receiver connects
    request_sender = Connect("inproc://requests", bind = True)
    result_receiver = Connect("inproc://results", bind = True)
    command_receiver = Connect("inproc://commands",
        options = {zmq.SUBSCRIBE : ""})  # subscribe to all command messages
    executor = ClusterExecutor(context, request_sender, result_receiver,
        command_sender, command_receiver, use_threading = True)
    executor.Setup()
    time.sleep(1)

    # launch the worker
    #   all sockets connect
    request_receiver = Connect("inproc://requests")
    result_sender = Connect("inproc://results")
    worker = threading.Thread(target = WorkerTarget, args = (context,
        request_receiver, result_sender, callback, command_receiver))
    worker.daemon = True
    worker.start()

    xs = range(10)
    expected = map(callback, xs)
    actual = sorted(ExecutorMap(executor, xs))
    self.assertEqual(expected, actual)

    # tear down the worker
    Worker.SendKillCommand(context, command_sender)
    worker.join(100)
    self.assertFalse(worker.is_alive())

    # tear down the cluster executor
    executor.Shutdown()
    self.assertFalse(executor.sink.is_alive())

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
