# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.models.viz2.cmds import EditParams, Imprint, Transform
from glimpse import util
from model import Model

def Main(pool, args):
  methods = dict((m.__name__, m) for m in (EditParams, Imprint, Transform))
  try:
    if len(args) < 1:
      raise util.UsageException()
    cmd = args[0]
    args = args[1:]
    if cmd not in methods:
      raise util.UsageException("Unknown command: %s" % (cmd,))
    methods[cmd](Model, pool, args)
  except util.UsageException, e:
    util.Usage("CMD\n%s" % "\n".join("  %s -- %s" % (name, method.__doc__)
        for name, method in methods.items()), e)
