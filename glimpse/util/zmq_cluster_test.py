
from itertools import islice
import logging
import threading
import time
import unittest
import zmq
from zmq_cluster import BasicVentilator, BasicSink, Ventilator, Sink, \
    ReceiverTimeoutException, Connect, ClusterResult, WorkerException, \
    BasicWorker

class TestCase(unittest.TestCase):

  def assertEmptyIter(self, it):
    try:
      it.next()
      self.fail("Expected empty iterator")
    except StopIteration:
      pass

class TestBasicVentilatorAndSink(TestCase):

  def _test_basicventilator(self, manual_setup):
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.bind("inproc://requests")
    receiver = context.socket(zmq.PULL)
    receiver.connect("inproc://requests")
    v = BasicVentilator(context, sender)
    if manual_setup:
      v.Setup()
    xs = range(10)
    num_sent = v.Send(xs)
    self.assertEqual(10, num_sent)
    ys = [ receiver.recv_pyobj() for _ in range(10) ]
    self.assertEqual(xs, ys)

  def test_basicventilator(self):
    self._test_basicventilator(manual_setup = True)

  def test_basicventilator_autosetup(self):
    self._test_basicventilator(manual_setup = False)

  def _test_basicsink(self, manual_setup):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("inproc://results")
    sender = context.socket(zmq.PUSH)
    sender.connect("inproc://results")
    s = BasicSink(context, receiver,
        receiver_timeout = 1000)  # set timeout to avoid infinite wait in case of
                                  # errors
    if manual_setup:
      s.Setup()
    xs = range(10)
    map(sender.send_pyobj, xs)
    ys_it = s.Receive(10)
    ys = list(islice(ys_it, 10))
    self.assertEmptyIter(ys_it)
    self.assertEqual(xs, ys)

  def test_basicsink(self):
    self._test_basicsink(manual_setup = True)

  def test_basicsink_autosetup(self):
    self._test_basicsink(manual_setup = False)

  def test_basicsink_timeout(self):
    def func():
      context = zmq.Context()
      receiver = context.socket(zmq.PULL)
      receiver.bind("inproc://results")
      sender = context.socket(zmq.PUSH)
      sender.connect("inproc://results")
      s = BasicSink(context, receiver)
      s.Setup()
      ys_it = s.Receive(timeout = 0)
      ys_it.next()
    self.assertRaises(ReceiverTimeoutException, func)

  def test_basicventilator_and_basicsink(self):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("inproc://results")
    sender = context.socket(zmq.PUSH)
    sender.connect("inproc://results")
    v = BasicVentilator(context, sender)
    s = BasicSink(context, receiver,
        receiver_timeout = 1000)  # set timeout to avoid infinite wait in case of
                                  # errors
    v.Setup()
    s.Setup()
    xs = range(10)
    num_sent = v.Send(xs)
    self.assertEqual(10, num_sent)
    ys_it = s.Receive(10)
    ys = list(islice(ys_it, 10))
    self.assertEmptyIter(ys_it)
    self.assertEqual(xs, ys)

  def test_basicsink_withquitcommand(self):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("inproc://results")
    sender = context.socket(zmq.PUSH)
    sender.connect("inproc://results")
    command_sender = context.socket(zmq.PUB)
    command_sender.bind("inproc://commands")
    command_receiver = context.socket(zmq.SUB)
    command_receiver.setsockopt(zmq.SUBSCRIBE, "")
    command_receiver.connect("inproc://commands")
    s = BasicSink(context, receiver, command_receiver,
        receiver_timeout = 1000)  # set timeout to avoid infinite wait in case of
                                  # errors
    s.Setup()
    xs = range(10)
    map(sender.send_pyobj, xs)
    ys_it = s.Receive()
    ys = list(islice(ys_it, 10))
    BasicSink.SendKillCommand(context, command_sender)
    self.assertEmptyIter(ys_it)
    self.assertEqual(xs, ys)

class TestVentilatorAndSink(TestCase):

  def test_sink(self):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("inproc://results")
    sender = context.socket(zmq.PUSH)
    sender.connect("inproc://results")
    s = Sink(context, receiver,
        receiver_timeout = 1000)  # set timeout to avoid infinite wait in case of
                                  # errors
    s.Setup()
    xs = range(10)
    xs2 = [ ClusterResult(status = ClusterResult.STATUS_SUCCESS, payload = x)
        for x in xs ]
    map(sender.send_pyobj, xs2)
    ys_it = s.Receive(10)
    ys = list(islice(ys_it, 10))
    self.assertEmptyIter(ys_it)
    self.assertEqual(xs, ys)

  # test that sink fails on status != STATUS_SUCCESS
  def test_sink_failsOnWorkerException(self):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("inproc://results")
    sender = context.socket(zmq.PUSH)
    sender.connect("inproc://results")
    s = Sink(context, receiver,
        receiver_timeout = 1000)  # set timeout to avoid infinite wait in case of
                                  # errors
    s.Setup()
    xs = range(10)
    xs2 = [ ClusterResult(status = ClusterResult.STATUS_FAIL, payload = x)
        for x in xs ]
    map(sender.send_pyobj, xs2)
    ys_it = s.Receive(10)
    def func():
      ys_it.next()
    self.assertRaises(WorkerException, func)

class Worker(BasicWorker):

  def __init__(self, context, request_receiver, result_sender, request_handler,
      command_receiver = None, receiver_timeout = None):
    super(Worker, self).__init__(context, request_receiver, result_sender,
        command_receiver, receiver_timeout)
    self.request_handler = request_handler

  def HandleRequest(self, request):
    result = self.request_handler(request)
    return result

def WorkerThreadTarget(ignore_timeout, context, request_receiver, result_sender,
    request_handler, command_receiver = None, receiver_timeout = None):
  worker = Worker(context, request_receiver, result_sender, request_handler,
      command_receiver, receiver_timeout)
  worker.Setup()
  try:
    worker.Run()
  except ReceiverTimeoutException:
    if not ignore_timeout:
      raise

class TestBasicWorker(TestCase):

  def test_worker_exitsOnCommand(self):
    context = zmq.Context()
    request_sender = context.socket(zmq.PUSH)
    request_sender.bind("inproc://requests")
    request_receiver = Connect(url = "inproc://requests")
    result_sender = Connect(url = "inproc://results")
    result_receiver = context.socket(zmq.PULL)
    result_receiver.bind("inproc://results")
    request_handler = lambda x: x * 100
    command_receiver = Connect(url = "inproc://commands")
    command_sender = context.socket(zmq.PUB)
    command_sender.bind("inproc://commands")
    receiver_timeout = 1000  # set timeout to avoid infinite wait in case of
                             # errors
    ignore_timeout = False  # timeout exceptions are a problem
    args = (ignore_timeout, context, request_receiver, result_sender,
        request_handler, command_receiver, receiver_timeout)
    worker = threading.Thread(target = WorkerThreadTarget, args = args)
    worker.daemon = True
    worker.start()
    Worker.SendKillCommand(context, command_sender)
    worker.join(1.5)  # wait 1 second for worker to timeout
    self.assertFalse(worker.is_alive())

  def test_worker_timesOutOnInactivity(self):
    context = zmq.Context()
    request_sender = context.socket(zmq.PUSH)
    request_sender.bind("inproc://requests")
    request_receiver = Connect(url = "inproc://requests")
    result_sender = Connect(url = "inproc://results")
    result_receiver = context.socket(zmq.PULL)
    result_receiver.bind("inproc://results")
    request_handler = lambda x: x * 100
    command_receiver = Connect(url = "inproc://commands")
    command_sender = context.socket(zmq.PUB)
    command_sender.bind("inproc://commands")
    receiver_timeout = 1  # set timeout to avoid infinite wait in case of errors
    worker = Worker(context, request_receiver, result_sender, request_handler,
        command_receiver, receiver_timeout)
    self.assertRaises(ReceiverTimeoutException, worker.Run)

  def test_worker(self):
    context = zmq.Context()
    request_sender = context.socket(zmq.PUSH)
    request_sender.bind("inproc://requests")
    request_receiver = Connect(url = "inproc://requests")
    result_sender = Connect(url = "inproc://results")
    result_receiver = context.socket(zmq.PULL)
    result_receiver.bind("inproc://results")
    request_handler = lambda x: x * 100
    command_receiver = Connect(url = "inproc://commands")
    command_sender = context.socket(zmq.PUB)
    command_sender.bind("inproc://commands")
    receiver_timeout = 100  # set timeout to avoid infinite wait in case of
                            # errors
    ignore_timeout = True  # let worker exit silently via recv() timeout
    args = (ignore_timeout, context, request_receiver, result_sender,
        request_handler, command_receiver, receiver_timeout)
    worker = threading.Thread(target = WorkerThreadTarget, args = args)
    worker.daemon = True
    worker.start()
    # Perform some tasks on the worker, and check results
    xs = range(10)
    map(request_sender.send_pyobj, xs)  # send requests to worker
    # Read results without blocking indefinitely
    poller = zmq.Poller()
    poller.register(result_receiver)
    actual = []
    for _ in range(10):
      socks = poller.poll(timeout = 10)
      self.assertNotEquals(0, len(socks))
      result = result_receiver.recv_pyobj()
      self.assertNotEqual(None, result)
      self.assertEquals(ClusterResult.STATUS_SUCCESS, result.status)
      actual.append(result.payload)
    expected = map(request_handler, xs)
    self.assertEqual(expected, actual)
    worker.join(1.5)  # wait long enough for worker to timeout
    self.assertFalse(worker.is_alive())

  def test_failsOnException(self):
    context = zmq.Context()
    request_sender = context.socket(zmq.PUSH)
    request_sender.bind("inproc://requests")
    request_receiver = Connect(url = "inproc://requests")
    result_sender = Connect(url = "inproc://results")
    result_receiver = context.socket(zmq.PULL)
    result_receiver.bind("inproc://results")
    def request_handler(x):
      if x:
        raise Exception
      return x
    command_receiver = Connect(url = "inproc://commands")
    command_sender = context.socket(zmq.PUB)
    command_sender.bind("inproc://commands")
    receiver_timeout = 100  # set timeout to avoid infinite wait in case of
                            # errors
    ignore_timeout = True  # let worker exit silently via recv() timeout
    args = (ignore_timeout, context, request_receiver, result_sender,
        request_handler, command_receiver, receiver_timeout)
    worker = threading.Thread(target = WorkerThreadTarget, args = args)
    worker.daemon = True
    worker.start()
    # Read results without blocking indefinitely
    poller = zmq.Poller()
    poller.register(result_receiver)
    # test that exceptions are caught and result has FAIL status
    request_sender.send_pyobj(True)  # trigger exception on worker
    socks = poller.poll(timeout = 10)
    self.assertNotEquals(0, len(socks))
    result = result_receiver.recv_pyobj()
    self.assertNotEqual(result, None)
    self.assertEquals(ClusterResult.STATUS_FAIL, result.status)
    # test that regular work can occur after exception is thrown
    request_sender.send_pyobj(False)  # trigger non-exception on worker
    socks = poller.poll(timeout = 10)
    self.assertNotEquals(0, len(socks))
    result = result_receiver.recv_pyobj()
    self.assertNotEqual(result, None)
    self.assertEquals(ClusterResult.STATUS_SUCCESS, result.status)
    expected = False
    actual = result.payload
    self.assertEqual(expected, actual)
    worker.join(1.5)  # wait long enough for worker to timeout
    self.assertFalse(worker.is_alive())

if __name__ == '__main__':

  # Uncomment the following for debugging messages.
  #~ logging.getLogger().setLevel(logging.INFO)

  # Uncomment the following to run a subset of tests.
  #~ suite = unittest.TestLoader().loadTestsFromTestCase(Test2)
  #~ unittest.TextTestRunner(verbosity=2).run(suite)

  unittest.main()
