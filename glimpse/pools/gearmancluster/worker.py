import gearman
import logging
import os
import socket
import threading
import time
import zmq

from .misc import *
from glimpse.pools import MulticorePool
from glimpse.util.zmq_cluster import FutureSocket, InitSocket

class Worker(gearman.GearmanWorker):
  """A Gearman worker that responds to an exit signal."""

  #: Exit status of the worker.
  exit_status = 0
  #: Indicates whether the worker has received a *quit* or *restart* command.
  finish = False
  #: How to marshal data on the wire.
  data_encoder = PickleDataEncoder
  #: Stateful memory for large, repeated request payloads.
  memory = dict()

  def after_poll(self, any_activity):
    """Returns True unless the :attr:`finish` flag is set.

    This method is called by the :class:`gearman.GearmanWorker` to decide when
    to quit.

    """
    return not self.finish

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
      cmd, payload = cmd_socket.recv_pyobj()
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
      elif cmd == COMMAND_SETMEMORY:
        logging.info("Gearman worker received SETMEMORY")
        if isinstance(payload, dict):
          memory = worker.memory
          for k, v in payload.items():
            logging.info("\tSETMEMORY[%s]" % k)
            memory[k] = v
        else:
          logging.warn("Invalid payload received for SETMEMORY command")
      elif cmd == COMMAND_CLEARMEMORY:
        logging.info("Gearman worker received CLEARMEMORY")
        worker.memory.clear()
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
  pool = MulticorePool(num_processes)
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
      if type(func) == int:
        logging.info("Looking up cached function for hash %d" % func)
        func = worker.memory.get(func, None)
      if func == None:
        raise Exception("No function found")
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
  logging.info("Gearman worker starting with %d cores and job server at %s" %
      (pool._processes, job_server_url))
  logging.info("\tSUB commands from %s" % command_url)
  logging.info("\tPUB logs to %s" % log_url)
  cmd_future = FutureSocket(url = command_url, bind = False)
  log_future = FutureSocket(url = log_url, bind = False)
  cmd_thread = threading.Thread(target = CommandHandlerTarget,
      args = (worker, cmd_future, log_future, get_stats))
  cmd_thread.daemon = True
  cmd_thread.start()
  # Start the task processor

  #~ worker.set_client_id("GlimpseWorker")

  worker.register_task(GEARMAN_TASK_MAP, handle_map)
  # Process tasks, checking command channel every two seconds
  worker.work(poll_timeout = 2.0)
  # Wait for worker to exit
  worker.shutdown()
  return worker.exit_status
