# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.executors.simple_cluster import LaunchBrokers, LaunchWorker, \
    KillWorkers, ClusterMap, ClusterConfig, ConfigException
from glimpse import util
import os
import sys

def MakeTransform():
  # use default model for now
  from glimpse.models import viz2
  from glimpse import backends
  model = viz2.Model(backends.CythonBackend(), viz2.Params())
  layer = viz2.Layer.IT
  return viz2.ModelTransform(model, layer, save_all = False)

def TransformImages(config, *fnames):
  cluster = ClusterMap(config)
  xform = MakeTransform()
  # Prepare files to be transmitted as transform requests
  from glimpse.models import InputSource
  from glimpse.models.viz2 import State
  input_states = map(State, map(InputSource, fnames))
  # Apply the transform by farming tasks to cluster worker nodes
  output_states = cluster.Map(xform, input_states)

def Methods():
  return map(eval, ("LaunchBrokers", "LaunchWorker", "KillWorkers"))

def main():
  try:
    config_files = tuple()
    opts, args = util.GetOptions("c:v")
    for opt, arg in opts:
      if opt == '-c':
        config_files = config_files + (arg,)
      elif opt == '-v':
        import logging
        logging.getLogger().setLevel(logging.INFO)
    if len(args) < 1:
      raise util.UsageException
    if not config_files:
      raise util.UsageException("Must specify a socket configuration file.")
    method = eval(args[0])
    config = ClusterConfig(*config_files)
    method(config, *args[1:])
  except ConfigException, e:
    sys.exit("Configuration error: %s" % e)
  except util.UsageException, e:
    methods = [ "  %s -- %s" % (m.func_name, m.__doc__.splitlines()[0])
        for m in Methods() ]
    util.Usage("[options] CMD [ARGS]\n"
        "  -c FILE   Read socket configuration from FILE\n"
        "CMDs include:\n" + "\n".join(methods),
        e)

if __name__ == "__main__":
  main()
