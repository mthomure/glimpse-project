# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# A command-line interface for managing a cluster of Glimpse workers.

import ConfigParser
from . import pool
from glimpse import util
from glimpse.util.zmq_cluster import MakeSocket
import logging
import os
import sys
import zmq

class ConfigException(Exception):
  """This exception indicates that an error occurred while reading the cluster
  configuration."""
  pass

def ReadClusterConfig(*config_files):
  config = ConfigParser.SafeConfigParser()
  read_files = config.read(config_files)
  if len(read_files) == 0:
    raise ConfigException("Unable to read any socket configuration files.")
  return config

def LaunchForwarder(config):
  """Start a set of intermediate devices for the cluster on this machine.
  RETURN This method does not return."""
  frontend_url = config.get('server', 'command_frontend_url')
  backend_url = config.get('server', 'command_backend_url')
  context = zmq.Context()
  cmd_be_socket = MakeSocket(context, url = backend_url, type = zmq.PUB,
      bind = False)
  cmd_fe_socket = MakeSocket(context, url = frontend_url, type = zmq.SUB,
      options = {zmq.SUBSCRIBE : ""})
  zmq.device(zmq.FORWARDER, cmd_fe_socket, cmd_be_socket)

def LaunchWorker(config, num_processes = None):
  """Start a cluster worker on the local host.
  config -- (ClusterConfig) socket configuration for the cluster
  num_processes -- (int) number of sub-processes to use
  RETURN This method calls sys.exit(), so does not return.
  """
  if num_processes != None:
    num_processes = int(num_processes)
  job_server_url = config.get('client', 'job_server_url')
  command_url = config.get('client', 'command_backend_url')
  exit_status = pool.RunWorker(job_server_url, command_url, num_processes)
  sys.exit(exit_status)

def SendCommand(command_url, command):
  context = zmq.Context()
  socket = context.socket(zmq.PUB)
  socket.connect(command_url)
  socket.send_pyobj(command)

def KillWorkers(config):
  """Send a QUIT command to any workers running on the cluster."""
  command_url = config.get('client', 'command_frontend_url')
  SendCommand(command_url, pool.COMMAND_QUIT)

def RestartWorkers(config):
  """Send KILL-ERROR to any workers running on cluster, causing a restart."""
  command_url = config.get('client', 'command_frontend_url')
  SendCommand(command_url, pool.COMMAND_RESTART)

def main():
  methods = map(eval, ("LaunchForwarder", "LaunchWorker", "KillWorkers",
      "RestartWorkers"))
  try:
    config_files = list()
    if 'GLIMPSE_CLUSTER_CONFIG' in os.environ:
      config_files.append(os.environ['GLIMPSE_CLUSTER_CONFIG'])
    opts, args = util.GetOptions("c:v")
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
    config = ReadClusterConfig(*config_files)
    method(config, *args[1:])
  except ConfigException, e:
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
