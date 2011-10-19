# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.backends.cython_backend import CythonBackend
from glimpse.backends.scipy_backend import ScipyBackend
from glimpse import util
from mutch_and_lowe import Model, Params
from viz2_cmds import MakeImprintHandler, MakeTransformHandler
import sys

def MakeModel(backend = "cython", params_fname = None, **args):
  mod_name = "%sBackend" % backend.capitalize()
  backend = eval("%s()" % mod_name)
  params = Params()
  if params_fname != None:
    params.LoadFromFile(params_fname)
  model = Model(backend, params, **args)
  return model

def PrintParamHelp():
  print "Possible values to set in the options file include:\n" + \
      "\n".join(("    %s - %s." % (k,v) for k,v in ALL_OPTIONS))
  sys.exit(-1)

def Main(args):
  try:
    if len(args) < 1:
      raise util.UsageException()
    cmd = args[0].lower()
    args = args[1:]
    if cmd == "imprint":
      Imprint = MakeImprintHandler(MakeModel, PrintParamHelp)
      Imprint(args)
    elif cmd == "transform":
      Transform = MakeTransformHandler(MakeModel, PrintParamHelp)
      Transform(args)
    else:
      raise util.UsageException("Unknown command: %s" % cmd)
  except util.UsageException, e:
    util.Usage("[imprint|transform]", e)
