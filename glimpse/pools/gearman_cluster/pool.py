# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import logging
import gearman
import ConfigParser
import cPickle
from glimpse import pools
from glimpse.util.zmq_cluster import FutureSocket, InitSocket
from glimpse import util
import itertools
import os
import socket
import threading
import time
import zmq

class PickleDataEncoder(gearman.DataEncoder):
  """A data encoder that reads and writes using the Pickle format.

  By default, a :class:`GearmanClient` can only send byte-strings. If we want to
  be able to send Python objects, we must specify a data encoder. This will
  automatically convert byte strings to Python objects (and vice versa) for
  *all* commands that have the 'data' field. See
  http://gearman.org/index.php?id=protocol for client commands that send/receive
  'opaque data'.

  """

  @classmethod
  def encode(cls, encodable_object):
    return cPickle.dumps(encodable_object, protocol = 2)

  @classmethod
  def decode(cls, decodable_string):
    return cPickle.loads(decodable_string)

#: Gearman task ID for the *map* task.
GEARMAN_TASK_MAP = "glimpse_map"
#: Instructs the worker to exit with signal 0.
COMMAND_QUIT = "quit"
#: Instructs the worker to exit with signal -1.
COMMAND_RESTART = "restart"
#: Instructs the worker to respond with processing statistics.
COMMAND_PING = "ping"

class Worker(gearman.GearmanWorker):
  """A Gearman worker that tracks exit status."""

  #: Exit status of the worker.
  exit_status = 0
  #: Indicates whether the worker has received a *quit* or *restart* command.
  finish = False
  #: How to marshal data on the wire.
  data_encoder = PickleDataEncoder

  def after_poll(self, any_activity):
    """Returns True unless the :attr:`finish` flag is set.

    This method is called by the :class:`gearman.GearmanWorker` to decide when
    to quit.

    """
    return not self.finish

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

  def map(self, func, iterable, chunksize = None):
    """Apply a function to a list."""
    if chunksize == None:
      chunksize = self.chunksize
    # chunk states into groups
    request_groups = util.GroupIterator(iterable, chunksize)
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
      failed_requests = [ r for r in job_requests if r.state == gearman.JOB_FAILED ]
      if len(failed_requests) > 0:
        buff = " First failed request: %s" % (failed_requests[0],)
      else:
        buff = ""
      raise Exception("Failed to process %d of %d tasks, %d timed out.%s" % \
          (len(failed_requests), len(completed),
          len(filter((lambda x: x.timed_out), job_requests)), buff))
    results = [ r.result for r in job_requests ]
    return list(util.UngroupIterator(results))

  #: Not implemented. This is currently an alias for :func:`map`.
  imap = map

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

  """
  config_files = list()
  if config_file != None:
    config_files.append(config_file)
  config_file = os.environ.get('GLIMPSE_CLUSTER_CONFIG', None)
  if config_file != None:
    config_files.append(config_file)
  config = ConfigParser.SafeConfigParser()
  read_files = config.read(config_files)
  if len(read_files) == 0:
    raise ConfigException("Unable to read any socket configuration files.")
  job_server_url = config.get('client', 'job_server_url')
  client = ClusterPool([job_server_url])
  if chunksize != None:
    client.chunksize = chunksize
  return client

def CommandHandlerTarget(worker, cmd_future, log_future, ping_handler):
  """Logic for process that handles command messages.

  A *quit* or *restart* command is handled by setting the worker's
  :attr:`finish <Worker.finish>` flag to True, setting the exit status, and
  returning. The exit status is 0 on *quit*, or -1 on *restart*.

  :param Worker worker: Instance for tracking worker state.
  :param FutureSocket cmd_future: Socket information for the command channel.
  :param FutureSocket log_future: Socket information for the log channel.
  :param callable ping_handler: Generates a response message for a *ping*
     command.

  """
  logging.info("Gearman command handler starting")
  context = zmq.Context()
  cmd_socket = InitSocket(context, cmd_future, type = zmq.SUB,
      options = {zmq.SUBSCRIBE : ""})
  log_socket = InitSocket(context, log_future, type = zmq.PUB)
  poller = zmq.Poller()
  poller.register(cmd_socket, zmq.POLLIN)
  while True:
    socks = dict(poller.poll())
    if cmd_socket in socks:
      cmd = cmd_socket.recv_pyobj()
      if cmd == COMMAND_QUIT:
        logging.info("Gearman worker quiting")
        # exit without error, so wrapper script will not restart worker
        worker.exit_status = 0
        worker.finish = True
        return
      elif cmd == COMMAND_RESTART:
        logging.info("Gearman worker restarting")
        # exit with error, so worker will be restarted
        worker.exit_status = -1
        worker.finish = True
        return
      elif cmd == COMMAND_PING:
        logging.info("Gearman worker received ping")
        stats = ping_handler()
        log_socket.send_pyobj(stats)
      else:
        logging.warn("Gearman worker got unknown command: %s" % (cmd,))

def RunWorker(job_server_url, command_url, log_url, num_processes = None):
  """Launch a Gearman worker and wait for it to complete.

  This worker processes batch requests using a :class:`MulticorePool
  <glimpse.pools.MulticorePool>`.

  :param str job_server_url: URL for Gearman job server.
  :param str command_url: URL for command channel.
  :param str log_url: URL for logging channel.
  :param int num_processes: Number of concurrent processes to use when
     processing a batch request. Defaults to the number of available cores.
  :returns: Exit status of the worker processes.

  """
  pool = pools.MulticorePool(num_processes)
  stats = dict(HOST = socket.getfqdn(), PID = os.getpid(), ELAPSED_TIME = 0,
      NUM_REQUESTS = 0, START_TIME = time.strftime("%Y-%m-%d %H:%M:%S"))
  get_stats = lambda: dict(stats.items())  # return a copy

  def handle_map(worker, job):
    """Map a function onto its arguments.
    worker -- (gearman.worker.GearmanWorker)
    job -- (gearman.job.GearmanJob)
    """
    try:
      start = time.time()
      func, args = job.data
      logging.info("Worker processing task with %d elements" % len(args))
      results = pool.map(func, args)
      elapsed_time = time.time() - start
      logging.info("\tfinished in %.2f secs" % elapsed_time)
      stats['ELAPSED_TIME'] += elapsed_time
      stats['NUM_REQUESTS'] += 1
      return results
    except Exception:
      logging.exception("Caught exception in worker")
      raise

  worker = Worker([job_server_url])
  # Start the command listener
  logging.info("Gearman worker starting with job server at %s" % job_server_url)
  logging.info("\tSUB commands from %s" % command_url)
  logging.info("\tPUB logs to %s" % log_url)
  cmd_future = FutureSocket(url = command_url, bind = False)
  log_future = FutureSocket(url = log_url, bind = False)
  cmd_thread = threading.Thread(target = CommandHandlerTarget,
      args = (worker, cmd_future, log_future, get_stats))
  cmd_thread.daemon = True
  cmd_thread.start()
  # Start the task processor
  worker.set_client_id("GlimpseWorker")
  worker.register_task(GEARMAN_TASK_MAP, handle_map)
  # Process tasks, checking command channel every two seconds
  worker.work(poll_timeout = 2.0)
  # Wait for worker to exit
  worker.shutdown()
  return worker.exit_status
