# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse import backends
from glimpse.models import viz2
from glimpse import util
import itertools

def Main(pool, args):
  try:
    if len(args) < 1:
      raise util.UsageException()
    cmd = args[0].lower()
    args = args[1:]
    if cmd == "imprint":
      Imprint(pool, args)
    elif cmd == "transform":
      Transform(pool, args)
    else:
      raise util.UsageException("Unknown command: %s" % cmd)
  except util.UsageException, e:
    util.Usage("[imprint|transform]", e)

def MakeInputStates(model, image_files, read_image_data):
  input_states = map(viz2.State, map(viz2.InputSource, image_files))
  if read_image_data:
    input_states = [ model.BuildLayer(state, viz2.Layer.IMAGE)
        for state in input_states ]
  return input_states

def Transform(pool, args):
  try:
    output_layer = viz2.Layer.C1
    save_all = False
    read_image_data = False
    opts, args = util.GetOptions("l:p:rs", args = args)
    s2_prototypes = None
    for opt, arg in opts:
      if opt == '-l':
        try:
          output_layer = viz2.Layer.FromName(arg)
        except ValueError:
          raise util.UsageException("Unknown model layer: %s" % arg)
      elif opt == '-p':
        s2_prototypes = util.Load(arg)
      elif opt == '-r':
        read_image_data = True
      elif opt == '-s':
        save_all = True
    if len(args) < 1:
      raise util.UsageException()
    model = viz2.Model(backends.CythonBackend(), viz2.Params())
    model.s2_kernels = s2_prototypes
    transform = viz2.ModelTransform(model, output_layer, save_all)
    input_states = MakeInputStates(model, args, read_image_data)
    # Map the transform over the image filenames.
    output_states = pool.imap_unordered(transform, input_states)
    map(util.Store, output_states)  # write state to stdout
  except util.UsageException, e:
    util.Usage("[options] IMAGE ... > RESULT-STREAM.dat\n"
        "  -l LAYR  Transform image through LAYR (r, s1, c1, s2, c2, it)"
        " (default: %s)\n" % output_layer + \
        "  -p FILE  Read S2 prototypes from FILE\n"
        "  -r       Read image data from disk before passing to map (only\n"
        "           useful for cluster pool)\n"
        "  -s       Save activity for all model layers, instead of just the \n"
        "           output layer"
        , e)

def Imprint(pool, args):
  try:
    save_locations = False
    normalize = True
    num_prototypes = 1
    read_image_data = False
    opts, args = util.GetOptions("ln:Nr", args = args)
    for opt, arg in opts:
      if opt == '-l':
        save_locations = True
      if opt == '-n':
        num_prototypes = int(arg)
      elif opt == '-N':
        normalize = False
      elif opt == '-r':
        read_image_data = True
    if len(args) < 1:
      raise util.UsageException()
    model = viz2.Model(backends.CythonBackend(), viz2.Params())
    sampler = viz2.C1PatchSampler(model, num_prototypes, normalize)
    input_states = MakeInputStates(model, args, read_image_data)
    # Map the sampler over the image filenames.
    protos_per_image = pool.imap_unordered(sampler, input_states)
    # We get a list of prototypes for each image. Chain them together.
    protos = itertools.chain(*protos_per_image)
    if save_locations:
      map(util.Store, protos)  # write patch activity and location to stdout
    else:
      for proto, location in protos:
        util.Store(proto)  # write only patch activity, using stdout
  except util.UsageException, e:
    util.Usage("[options] IMAGE ... > RESULT-STREAM.dat\n"
        "  -l       Save location of C1 patches, along with model activity\n"
        "  -n NUM   Imprint NUM prototypes (default: %d)\n" % num_prototypes + \
        "  -r       Read image data from disk before passing to map (only \n"
        "           useful for cluster pool)\n"
        "  -N       Disable normalization of C1 patches"
        , e)
