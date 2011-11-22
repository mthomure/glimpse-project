# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse import util
import itertools
import logging

class IExecutor:
  """An executor is an interface for interacting with a set of task processors.
  This interface includes a request queue (of which only the Put() method is
  exposed), and a result queue (of which only the Get() method is exposed)."""

  def Put(self, request):
    """Submit a task to the executor, whose result can be obtained by calling
    Get(). Each reques submitted via Put() is guaranteed to produce exactly one
    result via Get()."""

  def PutMany(self, requests):
    """Submit a batch of multiple requests. Some executors may have better
    performance in this case, as compared to submitting multiple individual
    requests.
    RETURN (int) the number of submitted requests"""

  def Get(self):
    """Retrieve the result for a request submitted via Put()."""

  def GetMany(self, num_results):
    """Retrieve a batch of results. As with PutMany(), some executors may have
    better performance using this method as compared to repeatedly calling
    Get().
    RETURN iterator over results. This iterator should be exhausted before any
           further calls to Get() or GetMany()."""

  def IsEmpty(self):
    """Determine if any results are available via Get()."""

class BasicExecutor:
  """A simple executor that applies a request handler to each request in the
  same thread."""

  def __init__(self, request_handler):
    self.request_handler = request_handler
    self.queue = []

  def Put(self, request):
    self.queue.append(self.request_handler(request))

  def PutMany(self, requests):
    return len(map(self.Put, requests))

  def Get(self):
    return self.queue.pop()

  def GetMany(self, num_results):
    return ( self.Get() for _ in range(num_results) )

  def IsEmpty(self):
    return len(self.queue) == 0

def StaticMap(executor, requests):
  """Apply the operation implemented by an executor across an iterable of
  requests. Note that the order of return elements may not match that of the
  input arguments (depending on implementation), and that an empty queue is
  assumed. This is considered a "static" map, since the request handler is
  specified when the executor is constructed.
  RETURN iterator over results. This iterator should be exhausted before any
         further calls to ExecutorMap(), or the Get()/GetMany() methods of the
         executor."""
  assert executor.IsEmpty(), "Map behavior is undefined unless executor " \
      "queue is empty."
  num_requests = executor.PutMany(requests)
  return executor.GetMany(num_requests)

def DynamicMap(executor, function, arguments, group_size = None):
  """Apply a user function to a set of arguments using the given executor. This
  is considered a "dynamic" map, since the request handler is specified at call
  time, and thus can change over the lifetime of the executor. As with
  StaticMap(), the order of output elements is not guaranteed.
  executor -- strategy for evaluating user function on elements. It is assumed
              that the executor applies DynamicMapRequestHandler() as its
              request handler function.
  function -- (callable) function that takes a single value as input, and
              returns a single value as output. Depending on the executor, note
              that the function may have to be picklable (e.g., it can not be a
              lambda function when using a MulticoreExecutor).
  arguments -- (iterable) input values for function
  RETURN iterator over resulting values. Returned iterator should be exhausted
         before any further calls to DynamicMap().
  """
  # First determine a useful request size for the executor, given the arguments.
  if group_size == None:
    num_arguments = None
    try:
      num_arguments = len(arguments)
    except TypeError:
      pass
    if hasattr(executor, 'GroupSize'):
      group_size = executor.GroupSize(num_arguments)

      logging.info("DynamicMap: executor (%s) chose group size of %d elements" % (executor, group_size))

      assert group_size > 0
    else:
      group_size = 1  # by default, make singleton requests
  # chunk states into groups
  request_groups = util.GroupIterator(arguments, group_size)
  # make tasks by combining each group with the transform
  batch_requests = itertools.izip(itertools.repeat(function), request_groups)
  # map executor across tasks
  result_groups = StaticMap(executor, batch_requests)
  # unchunk result groups
  return util.UngroupIterator(result_groups)

def DynamicMapRequestHandler(request):
  """This function should be passed to the executor's worker. It will be called
  by the zmq_cluster.Worker.Run() method when a new batch request (i.e., a
  request created by DynamicMap()) is available."""
  function, arguments = request
  results = map(function, arguments)
  import logging
  logging.info("DynamicMapRequestHandler [%d arguments]" % len(arguments))
  logging.info("  function: %r" % function)
  logging.info("  arguments: %r" % (arguments,))
  logging.info("  results: %r" % (results,))
  return results
