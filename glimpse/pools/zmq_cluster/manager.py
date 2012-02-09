# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.util import zmq_cluster
from glimpse.util.zmq_cluster import Connect, ReceiverTimeoutException
import itertools
import logging
import multiprocessing
import os
import threading
import time
import zmq

def _SinkTarget(out_queue, result_receiver, command_receiver = None,
    result_modifier = None, context = None, receiver_timeout = None):
  """Waits for worker results, applies a function to each result, and posts the
  output to shared queue.
  context -- (zmq.Context, optional) this is set for threaded sinks only
  receiver_timeout -- (int) time to wait for a result before quiting
  """
  # TODO: figure out how to pass a WorkerException back to the main process.
  logging.info("SinkTarget: starting up on pid %d" % os.getpid())
  if context == None:
    context = zmq.Context()
  sink = zmq_cluster.Sink(context, result_receiver,
      command_receiver = command_receiver, receiver_timeout = receiver_timeout)
  sink.Setup()
  logging.info("SinkTarget: about to receive")
  # Get an iterator over results. this will terminate when sink gets QUIT
  # command.
  results = sink.Receive(metadata = True)
  if result_modifier != None:
    results = itertools.imap(result_modifier, results)
  logging.info("SinkTarget: writing to queue")
  idx = 0
  # Walk the iterator returned by sink.Receive()
  for result in results:
    #~ logging.info("SinkTarget: got #%d result" % idx)
    out_queue.put(result)
    idx += 1
  #~ map(out_queue.put, results)
  logging.info("SinkTarget: shutting down on pid %d" % os.getpid())

class ClusterManager(object):
  """Applies a pre-defined operation to a set of elements in parallel, using
  a cluster of worker nodes. The operation is defined when the worker processes
  are launched. The user function is not supplied to the ClusterManager.
  Instead, it is assumed that all worker nodes use a zmq_cluster.Worker object
  to run the same request handler. In the sense that this object can be thought
  of as a client of the worker nodes, it is assumed that there is at most one
  client using the cluster at any given time."""

  SENTINEL = "DONE"

  def __init__(self, context, request_sender, result_receiver, command_sender,
      command_receiver, result_modifier = None, use_threading = False,
      receiver_timeout = None):
    """Create a new object.
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
    use_threading -- (bool) whether sink messages should be handled on a
                     seperate thread, rather than a sub-process.
    receiver_timeout -- (int) time to wait for a result before quiting
    """
    self.request_sender = request_sender
    self.result_receiver = result_receiver
    self.command_sender = command_sender
    self.command_receiver = command_receiver
    self.result_modifier = result_modifier
    self.context = context
    self.use_threading = use_threading
    self.receiver_timeout = receiver_timeout
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
    kwargs = {"receiver_timeout" : self.receiver_timeout}
    if self.use_threading:
      logging.info("ClusterManager: starting sink as thread")
      # Share context with threaded sink (required to avoid "connection denied"
      # errors).
      kwargs["context"] = self.context
      self.sink = threading.Thread(target = _SinkTarget, args = args,
          kwargs = kwargs, name = "SinkThread")
    else:
      logging.info("ClusterManager: starting sink as process")
      # Do not share context with sub-process sink (this would be an error).
      self.sink = multiprocessing.Process(target = _SinkTarget, args = args,
          name = "SinkProcess")
    self.sink.daemon = True
    self.sink.start()
    time.sleep(1)   # Sleep for one second to give existing workers time to
                    # connect.

  def Shutdown(self):
    # Signal the sink to quit
    if self.sink != None:
      if self.use_threading:
        logging.info("ClusterManager: sending kill command to sink thread")
      else:
        logging.info("ClusterManager: sending kill command to sink process")
      zmq_cluster.Sink.SendKillCommand(self.context, self.command_sender)
      logging.info("ClusterManager: waiting for sink to exit")
      timeout = 100
      self.sink.join(timeout)  # give the sink time to respond
      if self.sink.is_alive():
        logging.warn("ClusterManager: Sink did not respond to quit command")
      logging.info("ClusterManager: sink exited successfully")

  def Put(self, request, metadata = None):
    """Submit a task to the cluster, whose result can be obtained by calling
    Get(). Each request submitted via Put() is guaranteed to produce exactly one
    result."""
    if self.ventilator == None:
      self.Setup()
    self.ventilator.Send([request], metadata = metadata)
    self.task_count += 1

  def PutMany(self, requests, metadata = None):
    """Submit a batch of multiple requests. This method may have better
    performance in most cases, as compared to submitting multiple individual
    requests.
    RETURN (int) the number of submitted requests"""
    if self.ventilator == None:
      self.Setup()
    logging.info("ClusterManager: submitting multiple requests")
    num_requests = self.ventilator.Send(requests, metadata = metadata)
    self.task_count += num_requests
    return num_requests

  def Get(self, metadata = False):
    """Retrieve the result for a request submitted via Put()."""
    result, request_metadata, result_metadata = self.out_queue.get()
    self.task_count -= 1
    if metadata:
      return result, request_metadata, result_metadata
    return result

  def GetMany(self, num_results, metadata = False):
    """Retrieve a batch of results. As with PutMany(), this method may have
    better performance as compared to repeatedly calling Get().
    RETURN iterator over results. This iterator should be exhausted before any
           further calls to Get() or GetMany()."""
    # Return an iterator over the corresponding number of elements in the sink.
    logging.info("ClusterManager: returning iterator over %d results" % \
        num_results)
    return ( self.Get(metadata = metadata) for _ in range(num_results) )

  def IsEmpty(self):
    """Determine if any results are available via Get()."""
    return self.task_count == 0
