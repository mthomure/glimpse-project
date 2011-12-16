# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# A command-line interface for managing a cluster of Glimpse workers.

from .misc import LaunchBrokers, LaunchWorker, KillWorkers, PingWorkers, \
    RestartWorkers, FlushCluster
from .config import ClusterConfig, ConfigException
from glimpse import util
import os
import sys

def main():
  methods = map(eval, ("LaunchBrokers", "LaunchWorker", "KillWorkers",
      "RestartWorkers", "PingWorkers", "FlushCluster"))
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
    config = ClusterConfig(*config_files)
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
