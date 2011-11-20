# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.util import zmq_cluster
import itertools
import logging
from multicore_executor import QueueIterator
import multiprocessing
import threading
import os
import time
import zmq

def _SinkTarget(out_queue, result_receiver, command_receiver = None,
    result_modifier = None, context = None):
  """Waits for worker results, applies a function to each result, and posts the
  output to shared queue.
  context -- (zmq.Context, optional) this is set for threaded sinks only
  """
  logging.info("SinkTarget: starting up on pid %d" % os.getpid())
  if context == None:
    context = zmq.Context()
  sink = zmq_cluster.Sink(context, result_receiver, command_receiver)
  sink.Setup()
  logging.info("SinkTarget: about to receive")
  # Get an iterator over results. this will terminate when sink gets QUIT
  # command.
  results = sink.Receive()
  if result_modifier != None:
    results = itertools.imap(result_modifier, results)
  logging.info("SinkTarget: writing to queue")
  idx = 0
  # Walk the iterator returned by sink.Receive()
  for result in results:
    logging.info("SinkTarget: got %d'th result: %r" % (idx, result))
    out_queue.put(result)
    idx += 1
  #~ map(out_queue.put, results)
  logging.info("SinkTarget: shutting down on pid %d" % os.getpid())

class ClusterExecutor(object):
  """Applies a pre-defined operation to a set of elements in parallel, using
  a cluster of worker nodes. The operation is defined when the worker processes
  are launched. In contrast to the MulticoreExecutor, the user function is not
  supplied to this object's constructor. Instead, the ClusterWorker() function
  should be called on each worker node, with the user function passed as an
  argument. In the sense that this object can be thought of as a client of the
  worker nodes, it is assumed that there is at most one client using the cluster
  at any given time."""

  SENTINEL = "DONE"

  def __init__(self, context, request_sender, result_receiver,
      command_sender, command_receiver, result_modifier = None,
      chunk_size = None, use_threading = False):
    """Create a new cluster executor.
    context -- (zmq.Context)
    request_sender -- (zmq.socket or Connect) channel for sending requests to
                      worker nodes
    result_receiver -- (Connect) channel for collecting results from worker
                       nodes. Must be a Connect (not a zmq.socket), since it is
                       passed to a seperate thread/process.
    command_sender -- (zmq.socket or Connect) channel used to send quit commands
                      to cluster sink.
    command_receiver -- (Connect) how the sink should listen for quit commands.
                        This must be a Connect object (not a zmq.socket), since
                        it will be passed to a seperate thread or process.
    result_modifier -- (callable, optional) takes a result, and returns a
                       modified result. This can be used to perform work on
                       results (e.g., saving them to disk) as they arrive on the
                       cluster sink sub-process. Note that this function should
                       check the status indicator of a result before processing
                       its payload, since this will not be done before the
                       modifier function is called.
    chunk_size -- (int) number of requests to bundle together into a single
                  cluster request message
    use_threading -- (bool) whether sink messages should be handled on a
                     seperate thread, rather than a sub-process.
    """
    if chunk_size == None:
      # By default, make requests large enough to occupy 16-core machine
      chunk_size = 16
    self.request_sender = request_sender
    self.result_receiver = result_receiver
    self.command_sender = command_sender
    self.command_receiver = command_receiver
    self.result_modifier = result_modifier
    self.chunk_size = chunk_size
    self.context = context
    self.use_threading = use_threading
    self.in_queue = multiprocessing.Queue()
    self.out_queue = multiprocessing.Queue()
    self.ventilator = None
    self.sink = None
    self.task_count = 0  # number of out-standing task requests

  def Setup(self):
    assert self.in_queue.empty() and self.out_queue.empty()
    self.ventilator = zmq_cluster.Ventilator(self.context, self.request_sender)
    self.ventilator.Setup()
    args = (self.out_queue, self.result_receiver, self.command_receiver,
        self.result_modifier)
    if self.use_threading:
      logging.info("ClusterExecutor: starting sink as thread")
      # Share context with threaded sink (required to avoid "connection denied"
      # errors).
      self.sink = threading.Thread(target = _SinkTarget, args = args,
          kwargs = {"context" : self.context}, name = "SinkThread")
    else:
      logging.info("ClusterExecutor: starting sink as process")
      # Do not share context with sub-process sink (this would be an error).
      self.sink = multiprocessing.Process(target = _SinkWrapper, args = args,
          name = "SinkProcess")
    self.sink.daemon = True
    self.sink.start()
    time.sleep(1)   # Sleep for one second to give existing workers time to
                    # connect.

  def Shutdown(self):
    # Signal the sink to quit
    if self.sink != None:
      if self.use_threading:
        logging.info("ClusterExecutor: sending kill command to sink thread")
      else:
        logging.info("ClusterExecutor: sending kill command to sink process")
      zmq_cluster.Sink.SendKillCommand(self.context, self.command_sender)
      logging.info("ClusterExecutor: waiting for sink to exit")
      timeout = 100
      self.sink.join(timeout)  # give the sink time to respond
      if self.sink.is_alive():
        logging.warn("ClusterExecutor: Sink did not respond to quit command")
      logging.info("ClusterExecutor: sink exited successfully")

  def Put(self, request):
    """Submit a task to the cluster, whose result can be obtained by calling
    Get()."""
    if self.ventilator == None:
      self.Setup()
    self.ventilator.Send([request])
    self.task_count += 1

  def PutMany(self, requests):
    if self.ventilator == None:
      self.Setup()
    logging.info("ClusterExecutor: submitting multiple requests")
    num_requests = self.ventilator.Send(requests)
    self.task_count += num_requests
    return num_requests

  def Get(self):
    """Retrieve the result for a task submitted via Put()."""
    result = self.out_queue.get()
    self.task_count -= 1
    return result

  def GetMany(self, num_results):
    # Return an iterator over the corresponding number of elements in the sink.
    logging.info("ClusterExecutor: returning iterator over %d results" % \
        num_results)
    return ( self.Get() for _ in range(num_results) )

  def IsEmpty(self):
    return self.task_count == 0
