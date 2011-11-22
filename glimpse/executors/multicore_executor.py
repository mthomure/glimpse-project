# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse import util
import logging
import multiprocessing

def QueueIterator(queue, sentinel):
  """Iterate over values that appear on a queue, stopping when a sentinel value
  is detected."""
  while True:
    value = queue.get()
    if value == sentinel:
      break
    yield value
  raise StopIteration

class MulticoreRequest(object):
  """A multi-core request, corresponding to a subset of input elements."""

  chunk_idx = None  # an identifier for the current subset of input elements
  payload = None  # a subset of the input elements

class MulticoreResult(object):
  """A multi-core result, corresponding to the output value of a callback when
  applied to a subset of input elements."""

  chunk_idx = None  # an identifier corresponding to the task's chunk index
  status = None  # whether the input elements were processed successfully
  exception = None  # exception that occurrred during processing, if any
  payload = None  # output corresponding to task's input elements. this will
                  # either be a list -- in the case of a map() operation -- or a
                  # scalar -- in the case of a reduce().

  STATUS_SUCCESS = "OK"  # indicates that request was processed successfully
  STATUS_FAIL = "FAIL"  # indicates that error occurred while processing request

class _Subprocess(multiprocessing.Process):
  """A process that reads from a queue of tasks, applies a request handler to
  subsets of those tasks, and writes the results to an output queue."""

  SENTINEL = "DONE"  # placed on the input queue to indicate to sub-processes
                     # that they should terminate.

  def __init__(self, request_handler, in_queue, out_queue):
    """Create a process that will apply a request handler to subsets of input
    elements.
    request_handler -- (callable) takes a sequence of input elements to process.
                       returns a corresponding sequence of output elements (in
                       the case of a map operation), or a scalar (in the case of
                       a reduce operation).
    in_queue -- queue of input element chunks
    out_queue -- queue of output element chunks
    sentinel -- value that will be placed on the input queue when list of input
                elements is exhausted
    """
    super(_Subprocess, self).__init__(name = "MulticoreExecutorSubprocess")
    self.request_handler, self.in_queue, self.out_queue = (request_handler,
        in_queue, out_queue)

  def run(self):
    for request in QueueIterator(self.in_queue, self.SENTINEL):
      result = MulticoreResult()
      result.chunk_idx = request.chunk_idx
      try:
        result.payload = self.request_handler(request.payload)
        result.status = MulticoreResult.STATUS_SUCCESS
      except Exception, e:
        result.exception = e
        result.status = MulticoreResult.STATUS_FAIL
      self.out_queue.put(result, block = False)

class MulticoreExecutor(object):
  """Apply a request handler to a set of elements in parallel, using multiple
  sub-processes."""

  def __init__(self, request_handler, num_processes = None):
    """
    request_handler -- (callable) takes a single input element to process, and
                       returns a corresponding output element.
    num_processes -- (int) number of sub-processes to use. defaults to number of
                     cores on machine.
    """
    if num_processes == None:
      num_processes = multiprocessing.cpu_count()
    self.request_handler = request_handler
    self.num_processes = num_processes
    self._ready = False
    self.in_queue = multiprocessing.Queue()
    self.out_queue = multiprocessing.Queue()
    self.processes = []
    self.Setup()
    self.task_count = 0  # number of tasks submitted, but not returned

  def GroupSize(self, num_requests = None):
    """Determine a useful request size for the multicore executor. This is used
    by dynamic_executor.DynamicExecutorMapper.
    num_requests -- (int) total number of available requests
    """
    if num_requests == None:
      group_size = 10  # use a group size of 10 by default
      logging.info("MulticoreExecutor.GroupSize: can't determine number of "
          "requests, using default of %d requests per core" % group_size)
      return group_size
    group_size, extra = divmod(num_requests, self.num_processes)
    if extra > 0:
      group_size += 1
    logging.info("MulticoreExecutor.GroupSize: distributing "
        "%d requests across %d cores with group size of %d" % (
        num_requests, self.num_processes, group_size))
    return group_size

  def Setup(self):
    assert self.in_queue.empty()
    assert self.out_queue.empty()
    # Create and start worker processes.
    logging.info("Launching sub-processes")
    self.processes = []
    for _ in range(self.num_processes):
      p = _Subprocess(self.request_handler, self.in_queue, self.out_queue)
      p.daemon = True
      p.start()
      self.processes.append(p)
    self._ready = True

  def Shutdown(self):
    """Signal sub-processes to exit, and wait for completion."""
    # Terminate the input list, thus killing sub-processes.
    for _ in range(self.num_processes):
      self.in_queue.put(_Subprocess.SENTINEL)
    # Wait for sub-processes to finish.
    for p in self.processes:
      p.join()
    self._ready = False

  def Put(self, element):
    """Submit a task to sub-processes, whose result can be obtained by calling
    Get()."""
    request = MulticoreRequest()
    request.payload = element
    self.in_queue.put(request)
    self.task_count += 1

  def Get(self):
    """Retrieve the result for a task submitted via Put()."""
    result = self.out_queue.get()
    if result.status != MulticoreResult.STATUS_SUCCESS:
      raise Exception("Caught exception in sub-process: %s" % \
          result.exception)
    self.task_count -= 1
    return result.payload

  def PutMany(self, requests):
    num_requests = 0
    for request in requests:
      self.Put(request)
      num_requests += 1
    return num_requests

  def GetMany(self, num_results):
    return ( self.Get() for _ in range(num_results) )

  def IsEmpty(self):
    return self.task_count == 0
