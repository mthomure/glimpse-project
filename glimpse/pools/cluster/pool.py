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

  def __init__(self, config = None, manager = None):
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
    self.manager = manager

  def map(self, func, iterable, chunksize = None):
    """Perform stable map. Block until all results are available, which are
    returned as a list."""
    raise NotImplementedError

  def imap(self, func, iterable, chunksize = None):
    """Perform stable map. Return immediately with iterable, whose next() method
    blocks until the corresponding element is available."""
    raise NotImplementedError

  def imap_unordered(self, func, iterable, chunksize = None):
    """Perform non-stable sort. Return immediately with iterable, whose next()
    method blocks until the corresponding element is available.
    func -- (callable) function to apply. must be picklable.
    iterable -- values to which function is applied.
    chunksize -- (int) size of cluster request
    """
    if chunksize == None:
      chunksize = 16  # by default, send enough work to occupy a 16-core machine
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
