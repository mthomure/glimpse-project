import logging
import gearman
import cPickle
from glimpse import pools
from glimpse.util.zmq_cluster import FutureSocket, InitSocket
import threading
import zmq
from glimpse import util
import itertools
import time

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

class Worker(gearman.GearmanWorker):
  exit_status = 0
  finish = False
  data_encoder = PickleDataEncoder

  def after_poll(self, any_activity):
    return not self.finish

class Client(gearman.GearmanClient):
  data_encoder = PickleDataEncoder
  chunksize = 8
  poll_timeout = 60.0  # wait up to one minute for job to complete
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
    job_requests = self.submit_multiple_jobs(batch_requests,
        background = False, wait_until_complete = True,
        max_retries = self.max_retries, poll_timeout = self.poll_timeout)
    completed = [ r.complete and r.state == gearman.JOB_COMPLETE
        for r in job_requests ]
    if not all(completed):
      raise Exception("Failed to process %d of %d tasks in job" % \
          (len(filter((lambda x: not x), completed)), len(completed)))
    results = [ r.result for r in job_requests ]
    return list(util.UngroupIterator(results))

class ClusterPool(Client):

  def __init__(self, config, chunksize = None):
    if chunksize == None:
      chunksize = Client.chunksize
    job_server_url = config.get('client', 'job_server_url')
    super(ClusterPool, self).__init__([job_server_url])
    self.chunksize = chunksize

def CommandHandlerTarget(worker, cmd_future):
  logging.info("Gearman command handler starting")
  context = zmq.Context()
  socket = InitSocket(context, cmd_future, type = zmq.SUB, bind = False,
      options = {zmq.SUBSCRIBE : ""})
  poller = zmq.Poller()
  poller.register(socket, zmq.POLLIN)
  while True:
    socks = dict(poller.poll())
    if socket in socks:
      cmd = socket.recv_pyobj()
      if cmd == COMMAND_QUIT:
        logging.info("Gearman worker quiting")
        # exit with error, so wrapper script will not restart worker
        worker.exit_status = -1
        worker.finish = True
        return
      elif cmd == COMMAND_RESTART:
        logging.info("Gearman worker restarting")
        # exit without error, so worker will be restarted
        worker.exit_status = 0
        worker.finish = True
        return
      else:
        logging.info("Gearman worker quiting")

def RunWorker(job_server_url, command_url, num_processes = None):
  pool = pools.MulticorePool(num_processes)

  def handle_map(worker, job):
    """Map a function onto its arguments.
    worker -- (gearman.worker.GearmanWorker)
    job -- (gearman.job.GearmanJob)
    """
    try:
      start = time.time()
      func, args = job.data
      logging.info("Gearman worker processing task with %d elements" % len(args))
      results = pool.map(func, args)
      logging.info("\tfinished in %.2f secs" % (time.time() - start))
      return results
    except Exception, e:
      logging.info("\tcaught exception: %s" % e)
      raise

  worker = Worker([job_server_url])
  # Start the command listener
  cmd_future = FutureSocket(url = command_url, bind = False)
  cmd_thread = threading.Thread(target = CommandHandlerTarget,
      args = (worker, cmd_future))
  cmd_thread.daemon = True
  cmd_thread.start()
  # Start the task processor
  worker.set_client_id("GlimpseWorker")
  worker.register_task(GEARMAN_TASK_MAP, handle_map)
  logging.info("Gearman worker starting at %s" % job_server_url)
  # Process tasks, checking command channel every two seconds
  worker.work(poll_timeout = 2.0)
  worker.shutdown()
  return worker.exit_status

def check_request_status(job_request):
  if job_request.complete:
    print "Job %s finished!  Result: %s - %s" % (job_request.job.unique,
        job_request.state, job_request.result)
  elif job_request.timed_out:
    print "Job %s timed out!" % job_request.unique
  elif job_request.state == JOB_UNKNOWN:
    print "Job %s connection failed!" % job_request.unique

# start job server as
#   gearmand -vvvvvvvv -l gearmand.log -d
