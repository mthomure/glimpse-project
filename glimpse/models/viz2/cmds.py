# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse import backends
from model import Model
from glimpse import util
import Image
import itertools
import numpy as np
import sys

def Transform(model_class, pool, args):
  """Apply a model to a set of images, storing the resulting layer activity."""
  try:
    edit_params = False
    output_layer = None
    params = None
    save_all = False
    read_image_data = False
    opts, image_files = util.GetOptions("el:o:p:rs", args = args)
    s2_prototypes = None
    for opt, arg in opts:
      if opt == '-e':
        edit_params = True
      elif opt == '-l':
        try:
          output_layer = model_class.Layer.FromName(arg)
        except ValueError:
          raise util.UsageException("Unknown model layer: %s" % arg)
      elif opt == '-o':
        params = util.Load(arg)
      elif opt == '-p':
        s2_prototypes = util.Load(arg)
      elif opt == '-r':
        read_image_data = True
      elif opt == '-s':
        save_all = True
    if output_layer == None:
      raise util.UsageException("Missing output layer")
    if len(image_files) < 1:
      raise util.UsageException("Missing input images")
  except util.UsageException, e:
    util.Usage("[options] IMAGE ... > RESULT-STREAM.dat\n"
        "  -e       Edit model options in a GUI\n"
        "  -l LAYR  Transform image through LAYR (r, s1, c1, s2, c2, it)"
        " (default: %s)\n" % output_layer + \
        "  -o FILE  Read model options from FILE\n"
        "  -p FILE  Read S2 prototypes from FILE\n"
        "  -r       Read image data from disk before passing to map (only\n"
        "           useful for cluster pool)\n"
        "  -s       Save activity for all model layers, instead of just the \n"
        "           output layer"
        , e)
  if params == None:
    params = model_class.Params()
  if edit_params:
    params.configure_traits()
  model = model_class(backends.CythonBackend(), params)
  model.s2_kernels = s2_prototypes
  builder = model.BuildLayerCallback(output_layer, save_all)
  # Construct the input states.
  if read_image_data:
    input_states = [ model.MakeStateFromImage(Image.open(f))
        for f in image_files ]
  else:
    input_states = map(model.MakeStateFromFilename, image_files)
  # Map the transform over the image filenames.
  output_states = pool.imap_unordered(builder, input_states)
  map(util.Store, output_states)  # write state to stdout

def Imprint(model_class, pool, args):
  """Imprint a collection of S2 prototypes from a set of input images."""
  try:
    edit_params = False
    params = None
    save_locations = False
    normalize = True
    num_prototypes = 1
    read_image_data = False
    opts, image_files = util.GetOptions("eln:No:r", args = args)
    for opt, arg in opts:
      if opt == '-e':
        edit_params = True
      elif opt == '-l':
        save_locations = True
      if opt == '-n':
        num_prototypes = int(arg)
        if num_prototypes <= 0:
          raise util.UsageException("Number of prototypes must be positive")
      elif opt == '-N':
        normalize = False
      elif opt == '-o':
        params = util.Load(arg)
      elif opt == '-r':
        read_image_data = True
    if len(image_files) < 1:
      raise util.UsageException()
  except util.UsageException, e:
    util.Usage("[options] IMAGE ... > RESULT-STREAM.dat\n"
        "  -e       Edit model options in a GUI\n"
        "  -l       Save location of C1 patches, along with model activity\n"
        "  -n NUM   Imprint NUM prototypes (default: %d)\n" % num_prototypes + \
        "  -N       Disable normalization of C1 patches\n"
        "  -o FILE  Read model options from FILE\n"
        "  -r       Read image data from disk before passing to map (only \n"
        "           useful for cluster pool)"
        , e)
  if params == None:
    params = model_class.Params()
  if edit_params:
    params.configure_traits()
  model = model_class(backends.CythonBackend(), params)
  # Construct the input states.
  if read_image_data:
    input_states = [ model.MakeStateFromImage(Image.open(f))
        for f in image_files ]
  else:
    input_states = map(model.MakeStateFromFilename, image_files)
  prototypes, locations = model.ImprintS2Prototypes(num_prototypes,
      input_states, normalize = normalize, pool = pool)
  if save_locations:
    # write patch activity and location to stdout
    util.Store((prototypes, locations))
  else:
    # write only patch activity, using stdout
    util.Store(prototypes)

def EditParams(model_class, pool, args):
  """Configure a set of model parameters using a GUI."""
  try:
    params = None
    output = sys.stdout
    opts, image_files = util.GetOptions("i:o:", args = args)
    for opt, arg in opts:
      if opt == '-i':
        params = util.Load(arg)
        if not isinstance(params, model_class.Params):
          raise util.UsageException("Parameter data loaded from file has wrong "
              "type. Expected %s, but got %s." % \
              (util.TypeName(model_class.Params), util.TypeName(params)))
      elif opt == '-o':
        output = open(arg, "wb")
  except util.UsageException, e:
    util.Usage("[options] IMAGE ... > RESULT-STREAM.dat\n"
        "  -i FILE  Read model parameters from FILE\n"
        "  -o FILE  Write model parameters to FILE (default is stdout)"
        , e)
  if params == None:
    params = model_class.Params()
  params.configure_traits()
  util.Store(params, output)

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
