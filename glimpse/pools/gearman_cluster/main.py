# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# A command-line interface for managing a cluster of Glimpse workers.

from . import pool
from glimpse import util
from glimpse.util.zmq_cluster import MakeSocket
import logging
import os
import pprint
import sys
import threading
import time
import zmq

def LaunchBroker(config):
  """Start a set of intermediate devices for the cluster on this machine.

  This method does not return.

  :param ConfigParser config: Configuration information for cluster.

  """
  cmd_frontend_url = config.get('server', 'command_frontend_url')
  cmd_backend_url = config.get('server', 'command_backend_url')
  log_frontend_url = config.get('server', 'log_frontend_url')
  log_backend_url = config.get('server', 'log_backend_url')
  def thread_target():
    context = zmq.Context()
    log_be_socket = MakeSocket(context, url = log_backend_url, type = zmq.PUB,
        bind = True)
    log_fe_socket = MakeSocket(context, url = log_frontend_url, type = zmq.SUB,
        bind = True, options = {zmq.SUBSCRIBE : ""})
    zmq.device(zmq.FORWARDER, log_fe_socket, log_be_socket)
  thread = threading.Thread(target = thread_target)
  thread.daemon = True
  logging.info("Launching logging forwarder -- clients should SUB to "
      "%s and PUB to %s" % (log_backend_url, log_frontend_url))
  thread.start()
  context = zmq.Context()
  cmd_be_socket = MakeSocket(context, url = cmd_backend_url, type = zmq.PUB,
      bind = True)
  cmd_fe_socket = MakeSocket(context, url = cmd_frontend_url, type = zmq.SUB,
      bind = True, options = {zmq.SUBSCRIBE : ""})
  logging.info("Launching command forwarder -- clients should SUB to "
      "%s and PUB to %s" % (cmd_backend_url, cmd_frontend_url))
  zmq.device(zmq.FORWARDER, cmd_fe_socket, cmd_be_socket)

def LaunchWorker(config, num_processes = None):
  """Start a cluster worker on the local host.

  This method does not return.

  :param ConfigParser config: Configuration information for cluster.
  :param int num_processes: Number of sub-processes to use.

  """
  if num_processes != None:
    num_processes = int(num_processes)
  job_server_url = config.get('client', 'job_server_url')
  command_url = config.get('client', 'command_backend_url')
  log_url = config.get('client', 'log_frontend_url')
  exit_status = pool.RunWorker(job_server_url, command_url, log_url,
      num_processes)
  sys.exit(exit_status)

def SendCommand(command_url, command):
  """Send an arbitrary command to all workers running on the cluster.

  :param str command_url: URL of command channel.
  :param command: Command to send to workers.

  """
  context = zmq.Context()
  socket = context.socket(zmq.PUB)
  socket.connect(command_url)
  socket.send_pyobj(command)

def KillWorkers(config):
  """Send a *quit* command to all workers running on the cluster.

  :param ConfigParser config: Configuration information for cluster.

  """
  command_url = config.get('client', 'command_frontend_url')
  SendCommand(command_url, pool.COMMAND_QUIT)

def RestartWorkers(config):
  """Send a *restart* command to any workers running on cluster.

  :param ConfigParser config: Configuration information for cluster.

  """
  command_url = config.get('client', 'command_frontend_url')
  SendCommand(command_url, pool.COMMAND_RESTART)

def _PingWorkers(config, wait_time = None):
  """A result generator for the *ping* command.

  :param ConfigParser config: Configuration information for cluster.
  :param int wait_time: Amount of time to wait for responses, in seconds.

  """
  if wait_time == None:
    wait_time = 1  # wait for one second by default
  log_url = config.get('client', 'log_backend_url')
  context = zmq.Context()
  log_socket = MakeSocket(context, url = log_url, type = zmq.SUB, bind = False,
      options = {zmq.SUBSCRIBE : ""})
  command_url = config.get('client', 'command_frontend_url')
  SendCommand(command_url, pool.COMMAND_PING)
  poll_timeout = 1000  # poll for one second
  wait_start_time = time.time()
  results = list()
  poller = zmq.Poller()
  poller.register(log_socket, zmq.POLLIN)
  while time.time() - wait_start_time < wait_time:
    # Wait for input, with timeout given in milliseconds
    if poller.poll(timeout = poll_timeout):
      stats = log_socket.recv_pyobj()
      yield stats

def PingWorkers(config, wait_time = None):
  """Determine the set of active workers and print results to console.

  :param ConfigParser config: Configuration information for cluster.
  :param int wait_time: Amount of time to wait for responses, in seconds.

  """
  if wait_time != None:
    wait_time = int(wait_time)
  rs_list = _PingWorkers(config, wait_time)
  rs = dict()
  for r in rs_list:
    host = r['HOST']
    if host not in rs:
      rs[host] = []
    rs[host].append(r)
  widths = (30, 6, 5, 5, 10, 19)
  if rs:
    fields = ['HOST', 'PID', 'NREQS', 'ETIME', 'MEAN_ETIME', 'STIME']
    fields = [ str(f).center(w) for f, w in zip(fields, widths) ]
    print " ".join(fields)
    print " ".join([ "-" * w for w in widths ])
  for host in sorted(rs.keys()):
    for r in rs[host]:
      pid, nreqs, etime, stime = r['PID'], r['NUM_REQUESTS'], r['ELAPSED_TIME'], r['START_TIME']
      if nreqs > 0:
        mean_etime = etime / nreqs
      else:
        mean_etime = etime
      etime = "%.1f" % etime
      mean_etime = "%.1f" % mean_etime
      fields = (host, pid, nreqs, etime, mean_etime, stime)
      fields = [ str(f).ljust(w) for f, w in zip(fields, widths) ]
      print " ".join(fields)

def main(args = None):
  """Entry point for the command line interface."""
  methods = map(eval, ("LaunchBroker", "LaunchWorker", "KillWorkers",
      "RestartWorkers", "PingWorkers"))
  try:
    config_files = list()
    if 'GLIMPSE_CLUSTER_CONFIG' in os.environ:
      config_files.append(os.environ['GLIMPSE_CLUSTER_CONFIG'])
    opts, args = util.GetOptions("c:v", args = args)
    for opt, arg in opts:
      if opt == '-c':
        config_files.append(arg)
      elif opt == '-v':
        import logging
        logging.getLogger().setLevel(logging.INFO)
    if len(args) < 1:
      raise util.UsageException
    if len(config_files) == 0:
      raise util.UsageException("Must specify a socket configuration file.")
    method = eval(args[0])
    config = pool.ReadClusterConfig(*config_files)
    method(config, *args[1:])
  except pool.ConfigException, e:
    sys.exit("Configuration error: %s" % e)
  except util.UsageException, e:
    method_info = [ "  %s -- %s" % (m.func_name, m.__doc__.splitlines()[0])
        for m in methods ]
    util.Usage("[options] CMD [ARGS]\n"
        "  -c FILE   Read socket configuration from FILE\n"
        "  -v        Be verbose with logging\n"
        "CMDs include:\n" + "\n".join(method_info),
        e)

if __name__ == "__main__":
  main()
