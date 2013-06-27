# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import ConfigParser
import gearman
import itertools
import time
import zmq

from .misc import *
from glimpse.util.glist import GroupIterator, UngroupIterator

def SendCommand(command_url, command, payload = None):
  """Send an arbitrary command to all workers running on the cluster.

  :param str command_url: URL of command channel.
  :param command: Command to send to workers.

  """
  context = zmq.Context()
  socket = context.socket(zmq.PUB)
  socket.connect(command_url)
  msg = (command, payload)
  socket.send_pyobj(msg)

class ClusterPool(gearman.GearmanClient):
  """Client for the Gearman task farm."""

  #: Encode communications as pickles.
  data_encoder = PickleDataEncoder
  #: Size of a map request (in number of input elements).
  chunksize = 8
  #: Wait one minute for each job to complete.
  poll_timeout = 60.0
  #: Resubmit a job up to three times.
  max_retries = 3
  #: Whether to send function objects via stateful data store by default.
  cache_functions = False

  def __init__(self, host_list = None, command_url = None):
    """Create a new cluster pool.

    :param host_list: Gearman job server URLs.
    :type host_list: list of str
    :param str command_url: ZMQ channel for command messages.

    """
    super(ClusterPool, self).__init__(host_list)
    self.command_context = zmq.Context()
    self.command_socket = self.command_context.socket(zmq.PUB)
    self.command_socket.connect(command_url)

  def map(self, func, iterable, chunksize = None, cache_functions = None):
    """Apply a function to a list.

    :param func: Callable to evaluate.
    :param iterable: Input list.
    :param int chunksize: Number of arguments to pack into a single request.
    :param bool cache_functions: Whether the function should be stored in the
       cluster's stateful data store (useful for large function objects).
    :returns: Function output for each value in input list.
    :rtype: list

    """
    if chunksize == None:
      chunksize = self.chunksize
    if cache_functions == None:
      cache_functions = self.cache_functions
    # chunk states into groups
    request_groups = GroupIterator(iterable, chunksize)
    if func == None:
      raise ValueError("Function must not be empty.")
    if cache_functions:
      func_hash = hash(func)
      self.SetMemory({func_hash : func})
      func = func_hash
    # make tasks by combining each group with the transform
    batch_requests = [ dict(task = GEARMAN_TASK_MAP, data = data)
        for data in itertools.izip(itertools.repeat(func), request_groups) ]
    # Waits for all jobs to complete, returning a gearman.job.GearmanJobRequest
    # object for each chunk.
    timeout = self.poll_timeout * len(batch_requests)
    job_requests = self.submit_multiple_jobs(batch_requests,
        background = False, wait_until_complete = True,
        max_retries = self.max_retries, poll_timeout = timeout)
    completed = [ r.complete for r in job_requests ]
    if not all(r.complete and r.state == gearman.JOB_COMPLETE
        for r in job_requests):
      failed_requests = [ r for r in job_requests
          if r.state == gearman.JOB_FAILED ]
      if len(failed_requests) > 0:
        buff = " First failed request: %s" % (failed_requests[0],)
      else:
        buff = ""
      raise Exception("Failed to process %d of %d tasks, %d timed out.%s" % \
          (len(failed_requests), len(completed),
          len(filter((lambda x: x.timed_out), job_requests)), buff))
    results = [ r.result for r in job_requests ]
    if cache_functions:
      self.SetMemory({func_hash : None})
    return list(UngroupIterator(results))

  #: Not implemented. This is currently an alias for :func:`map`.
  imap = map

  def SetMemory(self, memory):
    """Set one or more variables in the cluster's stateful data store.

    :param dict memory: Variables to store.

    """
    msg = (COMMAND_SETMEMORY, memory)
    self.command_socket.send_pyobj(msg)
    # Wait for workers to process message
    time.sleep(5)

  def ClearMemory(self):
    """Remove all values from the cluster's stateful data store."""
    msg = (COMMAND_CLEARMEMORY, None)
    self.command_socket.send_pyobj(msg)
    # Wait for workers to process message
    time.sleep(5)

class ConfigException(Exception):
  """Indicates that an error occurred while reading the cluster
  configuration."""
  pass

def MakePool(config_file = None, chunksize = None):
  """Create a new client for a Gearman cluster.

  :param str config_file: Path to a configuration file.
  :param int chunksize: Size of batch request.
  :rtype: :class:`ClusterPool <glimpse.pools.gearman_pool.pool.ClusterPool>`

  Configuration information is read from the paths specified by
  ``config_file`` (if specified), and the ``GLIMPSE_CLUSTER_CONFIG``
  environment variable (if set).

  Example config file:

     [client]
     host = belmont.cs.pdx.edu
     command_backend_url = tcp://%(host)s:9001
     command_frontend_url = tcp://%(host)s:9000
     log_backend_url = tcp://%(host)s:9003
     log_frontend_url = tcp://%(host)s:9002
     job_server_url = %(host)s:4730
     cache_functions = True
     chunksize = 8
     poll_timeout = 60.0
     max_retries = 3

  """
  config_files = list()
  if config_file != None:
    config_files.append(config_file)
  config = ConfigParser.SafeConfigParser()
  read_files = config.read(config_files)
  if len(read_files) == 0:
    raise ConfigException("Unable to read any socket configuration files.")
  job_server_url = config.get('client', 'job_server_url')
  command_url = config.get('client', 'command_frontend_url')
  client = ClusterPool(host_list = [job_server_url], command_url = command_url)
  if config.has_option('client', 'chunksize'):
    chunksize = config.getint('client', 'chunksize')
  if chunksize != None:
    client.chunksize = chunksize
  if config.has_option('client', 'poll_timeout'):
    client.poll_timeout = config.getfloat('client', 'poll_timeout')
  if config.has_option('client', 'max_retries'):
    client.max_retries = config.getint('client', 'max_retries')
  if config.has_option('client', 'cache_functions'):
    client.cache_functions = config.getboolean('client', 'cache_functions')
  return client
