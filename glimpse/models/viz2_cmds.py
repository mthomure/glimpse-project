# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.backends.cython_backend import CythonBackend
from glimpse.backends.scipy_backend import ScipyBackend
import Image
import os
import sys
from viz2 import *

def MakeModel(backend = "cython", params_fname = None, **args):
  mod_name = "%sBackend" % backend.capitalize()
  backend = eval("%s()" % mod_name)
  params = Viz2Params()
  if params_fname != None:
    params.LoadFromFile(params_fname)
  model = Viz2Model(backend, params, **args)
  return model

def PrintParamHelp():
  print "Possible values to set in the options file include:\n" + \
      "\n".join(("    %s - %s." % (k,v) for k,v in ALL_OPTIONS))
  sys.exit(-1)

def MakeImprintHandler(model_func, param_help_func):
  def Imprint(args):
    backend = "cython"
    num_prototypes = 1
    try:
      params_fname = None
      print_locations = False
      stream = False
      opts, args = util.GetOptions("b:Hln:o:p:s", args = args)
      for opt, arg in opts:
        if opt == '-b':
          backend = arg
        elif opt == '-H':
          param_help_func()
        elif opt == '-l':
          print_locations = True
        elif opt == '-n':
          num_prototypes = int(arg)
        elif opt == '-o':
          params_fname = arg
        elif opt == '-s':
          stream = True
      if len(args) < 1:
        raise util.UsageException()
      all_protos = []
      all_locations = []
      for ifname in args:
        img = Image.open(ifname)
        model = model_func(backend, params_fname)
        protos, locations = model.ImprintPrototypes(img, num_prototypes)
        all_protos.extend(protos)
        all_locations.extend(locations)
      if stream == True:
        for p in all_protos:
          util.Store(p, sys.stdout)
      else:
        util.Store(np.array(all_protos), sys.stdout)
      if print_locations == True:
        for loc in all_locations:
          print >>sys.stderr, ifname + " " + " ".join(map(str, loc))
    except util.UsageException, e:
      if e.msg:
        print >>sys.stderr, e.msg
      util.Usage("[options] IMAGE ...\n"
          "  -b STR   Set backend type (one of cython, scipy."
          " default: %s)\n" % backend + \
          "  -h       Print this help and exit\n"
          "  -H       Print extra help regarding valid options, and exit\n"
          "  -l       Print location of C1 patches used as prototypes\n"
          "  -n INT   Set number of prototypes to imprint"
          " (default: %d)\n" % num_prototypes + \
          "  -o PATH  Path to options file\n"
          "  -s       Print prototypes as a stream"
      )
  return Imprint

# Identifiers for objects that can be stored. The 'feature-vector' is currently
# a copy of the IT activity. In the future, however, this may be a concatenation
# of C1, C2, and IT activity.
STORAGE_ELEMENTS = set(['options', 'image', 's1-kernels', 's2-kernels',
    'feature-vector'] + [ "%s-activity" % x for x in ALL_LAYERS ])

def MakeTransformHandler(model_func, param_help_func):
  def Transform(args):
    backend = "cython"
    store_list = "none"
    output_layer = LAYER_IT
    try:
      params_fname = None
      proto_fname = None
      rdir = None
      print_features = False
      protos = None
      opts, args = util.GetOptions("b:fHl:o:p:r:s:", args = args)
      for opt, arg in opts:
        if opt == '-b':
          backend = arg
        elif opt == '-f':
          print_features = True
        elif opt == '-H':
          param_help_func()
        if opt == '-l':
          if arg not in ALL_LAYERS:
            raise util.UsageException("Invalid layer (-l) name: %s" % arg)
          output_layer = arg
        elif opt == '-o':
          params_fname = arg
        elif opt == '-p':
          protos = util.Load(arg)
        elif opt == '-r':
          rdir = arg
        elif opt == '-s':
          store_list = arg
      if len(args) < 1:
        raise util.UsageException()
      if store_list == "all":
        store_list = STORAGE_ELEMENTS
      elif store_list == "none":
        store_list = set()
      else:
        store_list = set(x.lower() for x in store_list.split(","))
        if not store_list.issubset(STORAGE_ELEMENTS):
          raise util.UsageException("User specified invalid storage (-s)" \
              "elements: %s" % ",".join(store_list.difference(STORAGE_ELEMENTS)))
      if output_layer in (LAYER_S2, LAYER_C2, LAYER_IT) and protos == None:
        raise util.UsageException("Must specify S2 prototypes to compute"
            " '%s' layer" % output_layer)
      if rdir != None:
        if len(args) > 1:
          raise util.UsageException("Can't specify result directory when "
              "transforming multiple images.")
        if not os.path.isdir(rdir):
          raise util.UsageException("Not an existing directory: %s" % rdir)
      for ifname in args:
        img = Image.open(ifname)
        model = model_func(backend, params_fname, s2_kernels = protos)
        results = model.BuildLayers(img, LAYER_IMAGE, output_layer)
        if rdir != None:
          results_for_output = dict([ ("%s-activity" % k, v) for k, v
              in results.items() ])
          results_for_output['options'] = model.params
          results_for_output['s1-kernels'] = model.s1_kernels
          results_for_output['s2-kernels'] = model.s2_kernels
          if LAYER_IT in results:
            results_for_output['feature-vector'] = results[LAYER_IT]
          for name in set(results_for_output.keys()).intersection(store_list):
            fname = os.path.join(rdir, name)
            util.Store(results_for_output[name], fname)
        if print_features:
          util.Store(results[LAYER_IT], sys.stdout)
    except util.UsageException, e:
      if e.msg:
        print >>sys.stderr, e.msg
      util.Usage("[options] IMAGE\n"
          "  -b STR   Set backend type (one of cython, scipy."
          " default: %s)\n" % backend + \
          "  -f       Write feature vector to stdout\n"
          "  -h       Print this help and exit\n"
          "  -H       Print extra help regarding valid options, and exit\n"
          "  -l LAYR  Transform image through LAYR (r, s1, c1, s2, c2, it)"
          " (default: %s)\n" % output_layer + \
          "  -o PATH  Path to options file\n"
          "  -p PATH  Path to S2 prototypes\n"
          "  -r PATH  Path to result directory\n"
          """  -s STR   Select layer information to be stored -- given by comma-
             seperated STR. Can also be the special values 'all' or 'none'.
             Legal values include:\n%s""" % \
              "\n".join(("             %s" % e \
                  for e in sorted(STORAGE_ELEMENTS)))
      )
  return Transform

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
