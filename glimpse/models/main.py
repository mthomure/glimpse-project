# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# This is a frontend script for model-specific operations.

from glimpse import pools
from glimpse import util
import logging
import sys

def main():
  default_model = "viz2"
  try:
    opts, args = util.GetOptions("c:m:v")
    model_name = default_model
    pool = None
    pool = pools.MulticorePool()
    for opt, arg in opts:
      if opt == '-c':
        # Use a cluster of worker nodes
        from glimpse.pools.cluster import ClusterConfig, ClusterPool
        config = ClusterConfig(arg)
        pool = ClusterPool(config)
      elif opt == '-m':
        # Set the model class
        arg = arg.lower()
        if arg == "default":
          model_name = default_model
        else:
          model_name = arg
      elif opt == '-v':
        logging.getLogger().setLevel(logging.INFO)
    models = __import__("glimpse.models.%s" % model_name, globals(), locals(),
        ['cmds'], 0)
    try:
      model = getattr(models, 'cmds')
    except AttributeError:
      raise util.UsageException("Unknown model (-m): %s" % model_name)
  except util.UsageException, e:
    util.Usage("[options]\n"
        "  -c FILE  Use a cluster to evaluate images, configured in FILE.\n"
        "  -m MOD   Use model named MOD\n"
        "  -v       Enable verbose logging",
        e
    )
  model.Main(pool, args)

if __name__ == '__main__':
  main()
