# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse import util
from glimpse.pools import MakePool
import itertools
import logging
from manager import ClusterManager
from glimpse.util.zmq_cluster import BasicWorker
import zmq

class ClusterPool(object):
  """This class is meant to be a port of the multiprocessing.Pool interface to
  the context of cluster computing."""

  def __init__(self, config = None, manager = None, chunksize = None):
    """Create a new object.
    config -- (ClusterConfig) a cluster configuration. Used to create a
              ClusterManager if none is given.
    manager -- (ClusterManager) the manager with which to process requests
    """
    if manager == None:
      assert config != None, "Must specify ClusterConfig or ClusterManager"
      context = zmq.Context()
      manager = ClusterManager(context, config.request_sender,
          config.result_receiver, config.command_sender,
          config.command_receiver)
    if chunksize == None:
      chunksize = 16  # by default, send enough work to occupy a 16-core machine
    self.manager = manager
    self.chunksize = chunksize
    self.cluster_stats = {}

  def map(self, func, iterable, chunksize = None):
    """Perform stable map. Block until all results are available, which are
    returned as a list."""
    return list(self.imap(func, iterable, chunksize))

  def imap(self, func, iterable, chunksize = None):
    """Perform stable map. Return immediately with iterable, whose next() method
    blocks until the corresponding element is available."""
    if chunksize == None:
      chunksize = self.chunksize
    # chunk states into groups
    request_groups = util.GroupIterator(iterable, chunksize)
    # make tasks by combining each group with the transform
    batch_requests = itertools.izip(itertools.repeat(func), request_groups)
    # map manager across tasks
    assert self.manager.IsEmpty()
    call_id = hash(func)
    # pass an infinite iterator of request IDs
    def metadata():
      x = 0
      while True:
        yield call_id, x
        x += 1
    num_requests = self.manager.PutMany(batch_requests, metadata = metadata())
    result_groups = self.manager.GetMany(num_requests, metadata = True)
    ordered_result_groups = [None] * num_requests
    for group in result_groups:
      results, request_metadata, result_metadata = group
      valid = True
      try:
        call_id_, offset = request_metadata
        if call_id_ != call_id:
          valid = False
      except:
        valid = False
      if not valid:
        raise ValueError("Got results for wrong request")
      self._HandleResultMetadata(result_metadata)
      ordered_result_groups[offset] = results
    # unchunk result groups -- this assumes workers maintain order of requets
    # within a batch
    return util.UngroupIterator(ordered_result_groups)

  def _HandleResultMetadata(self, result_metadata):
    fqdn, pid, elapsed_time = result_metadata
    key = (fqdn, pid)
    if key not in self.cluster_stats:
      self.cluster_stats[key] = []
    self.cluster_stats[key].append(elapsed_time)

  def imap_unordered(self, func, iterable, chunksize = None):
    """Perform non-stable sort. Return immediately with iterable, whose next()
    method blocks until the corresponding element is available.
    func -- (callable) function to apply. must be picklable.
    iterable -- values to which function is applied.
    chunksize -- (int) size of cluster request
    """
    if chunksize == None:
      chunksize = self.chunksize
    # chunk states into groups
    request_groups = util.GroupIterator(iterable, chunksize)
    # make tasks by combining each group with the transform
    batch_requests = itertools.izip(itertools.repeat(func), request_groups)
    # map manager across tasks
    assert self.manager.IsEmpty()
    num_requests = self.manager.PutMany(batch_requests)
    result_groups = self.manager.GetMany(num_requests)
    # unchunk result groups
    return util.UngroupIterator(result_groups)

class PoolWorker(BasicWorker):

  def __init__(self, context, config, receiver_timeout = None, pool = None):
    super(PoolWorker, self).__init__(context, config.request_receiver,
      config.result_sender, command_receiver = config.command_receiver,
      receiver_timeout = receiver_timeout)
    if pool == None:
      pool = MakePool()
    self.pool = pool

  def HandleRequest(self, dynamic_batch_request):
    """Convert a dynamic batch request to a batch result."""
    function, batch_request = dynamic_batch_request
    logging.info("PoolWorker: mapping function across "
        "%d elements" % len(batch_request))
    result_list = self.pool.map(function, batch_request)
    return result_list  # return batch result
