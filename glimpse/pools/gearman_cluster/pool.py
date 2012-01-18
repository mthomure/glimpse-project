import logging
import gearman
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

# By default, GearmanClient's can only send off byte-strings. If we want to be
# able to send out Python objects, we can specify a data encoder. This will
# automatically convert byte strings <-> Python objects for ALL commands that
# have the 'data' field. See http://gearman.org/index.php?id=protocol for client
# commands that send/receive 'opaque data'.
class PickleDataEncoder(gearman.DataEncoder):

  @classmethod
  def encode(cls, encodable_object):
    return cPickle.dumps(encodable_object, protocol = 2)

  @classmethod
  def decode(cls, decodable_string):
    return cPickle.loads(decodable_string)

GEARMAN_TASK_MAP = "glimpse_map"
COMMAND_QUIT = "quit"
COMMAND_RESTART = "restart"
COMMAND_PING = "ping"

class Worker(gearman.GearmanWorker):
  exit_status = 0
  finish = False
  data_encoder = PickleDataEncoder

  def after_poll(self, any_activity):
    return not self.finish

class Client(gearman.GearmanClient):
  data_encoder = PickleDataEncoder
  chunksize = 8
  poll_timeout = 60.0  # wait one minute for each job to complete
  max_retries = 3  # resubmit job up to three times

  def map(self, func, iterable, chunksize = None):
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
          (len(filter((lambda x: not x), completed)), len(completed),
          len(filter((lambda x: x.timed_out), job_requests)), buff))
    results = [ r.result for r in job_requests ]
    return list(util.UngroupIterator(results))

class ClusterPool(Client):

  def __init__(self, config, chunksize = None):
    if chunksize == None:
      chunksize = Client.chunksize
    job_server_url = config.get('client', 'job_server_url')
    super(ClusterPool, self).__init__([job_server_url])
    self.chunksize = chunksize

def CommandHandlerTarget(worker, cmd_future, log_future, ping_handler):
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
    except Exception, e:
      logging.warn("\tcaught exception: %s" % e)
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
  worker.shutdown()
  return worker.exit_status

# start job server as
#   gearmand -vvvvvvvv -l gearmand.log -d
