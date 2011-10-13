# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# Create a 2-part, HMAX-like hierarchy of S+C layers.
# This module implements the "Viz2" model used for the GCNC 2011 experiments.

from glimpse import util
from glimpse.util import ACTIVATION_DTYPE
from glimpse.util import kernel
import numpy as np
import random

def PrototypeSampler(c1s, num_prototypes, kwidth, scale_norm):
  """
  c1 - list of (3-D) C1 activity maps, one map per scale.
  RETURNS: iterator over prototype arrays. Note that returned prototypes should
           be copied, as the returned array may be reused.
  """
  assert (np.array(len(c1s[0].shape)) == 3).all()
  num_scales = len(c1s)
  num_orientations = len(c1s[0])
  for i in range(num_prototypes):
    scale = random.randint(0, num_scales - 1)
    c1 = c1s[scale]
    height, width = c1.shape[-2:]
    y = random.randint(0, height - kwidth)
    x = random.randint(0, width - kwidth)
    proto = c1[ :, y : y + kwidth, x : x + kwidth ]
    proto = proto.copy()
    if scale_norm:
      util.ScaleUnitNorm(proto)
    yield proto, (scale, y, x)
  raise StopIteration

class Viz2Model(object):

  def __init__(self, backend, params, s1_kernels = None, s2_kernels = None):
    self.backend = backend
    self.params = params
    if s1_kernels == None:
      s1_kernels = kernel.MakeMultiScaleGaborKernels(
          kwidth = params['s1_kwidth'], num_scales = params['num_scales'],
          num_orientations = params['s1_num_orientations'],
          num_phases = params['s1_num_phases'], shift_orientations = True,
          scale_norm = True)
    self.s1_kernels = s1_kernels
    self.s2_kernels = s2_kernels

  def BuildRetinaFromImage(self, img):
    if not self.params['retina_enabled']:
      return img
    retina = self.backend.ContrastEnhance(img,
        kwidth = self.params['retina_kwidth'],
        bias = self.params['retina_bias'],
        scaling = 1)
    return retina

  def BuildS1FromRetina(self, retina):
    # Reshape retina to be 3D array
    retina_ = retina.reshape((1,) + retina.shape)
    num_scales = self.params['num_scales']
    s1_kwidth = self.params['s1_kwidth']
    s1_num_orientations = self.params['s1_num_orientations']
    s1_num_phases = self.params['s1_num_phases']
    # Reshape kernel array to be 4-D: scale, index, 1, y, x
    s1_kernels = self.s1_kernels.reshape((num_scales, -1, 1, s1_kwidth,
        s1_kwidth))
    s1s = []
    for scale in range(num_scales):
      ks = s1_kernels[scale]
      s1_ = self.backend.NormRbf(retina_, ks, bias = self.params['s1_bias'],
          beta = self.params['s1_beta'], scaling = self.params['s1_scaling'])
      # Reshape S1 to be 4D array
      s1 = s1_.reshape((s1_num_orientations, s1_num_phases) + s1_.shape[-2:])
      s1 = s1.max(1)
      s1s.append(s1)
    return s1s

  def BuildC1FromS1(self, s1s):
    num_scales = self.params['num_scales']
    c1s = [ self.backend.LocalMax(s1, kwidth = self.params['c1_kwidth'],
        scaling = self.params['c1_scaling']) for s1 in s1s ]
    # DEBUG: use this to whiten over orientation only
    #~ if self.params['c1_whiten']:
      #~ for c1 in c1s:
        #~ Whiten(c1)
    # DEBUG: use this to whiten over scale AND orientation concurrently. PANN
    # and the old Viz2 model used this.
    if self.params['c1_whiten']:
      c1s = np.array(c1s, ACTIVATION_DTYPE)
      c1_shape = c1s.shape
      c1s = c1s.reshape((-1,) + c1s.shape[-2:])
      Whiten(c1s)
      c1s = c1s.reshape(c1_shape)
    return c1s

  def BuildS2FromC1(self, c1s):
    CheckPrototypes(self.s2_kernels)
    num_scales = self.params['num_scales']
    s2s = []
    for scale in range(num_scales):
      c1 = c1s[scale]
      s2 = self.backend.NormRbf(c1, self.s2_kernels,
          bias = self.params['s2_bias'], beta = self.params['s2_beta'],
          scaling = self.params['s2_scaling'])
      s2s.append(s2)
    return s2s

  def BuildC2FromS2(self, s2s):
    c2s = map(self.backend.GlobalMax, s2s)
    return c2s

  def BuildItFromC2(self, c2s):
    it = np.array(c2s).max(0)
    return it

  def BuildLayers(self, data, input_layer, output_layer):
    start = ALL_LAYERS.index(input_layer)
    stop = ALL_LAYERS.index(output_layer)
    layers = ALL_LAYERS[start : stop + 1]
    results = dict()
    for idx in range(start, stop + 1):
      builder = ALL_BUILDERS[idx]
      data = builder(self, data)
      layer_name = ALL_LAYERS[idx]
      results[layer_name] = data
    return results

  def ImprintPrototypes(self, img, num_prototypes):
    """Compute C1 activity maps and sample patches from random locations.
    RETURNS list of prototypes, and list of corresponding locations.
    """
    results = self.BuildLayers(img, LAYER_RETINA, LAYER_C1)
    c1s = results[LAYER_C1]
    proto_it = PrototypeSampler(c1s, num_prototypes,
        kwidth = self.params['s2_kwidth'], scale_norm = True)
    protos = list(proto_it)
    return zip(*protos)

# Identifiers for layers that can be computed
LAYER_RETINA = 'r'
LAYER_S1 = 's1'
LAYER_C1 = 'c1'
LAYER_S2 = 's2'
LAYER_C2 = 'c2'
LAYER_IT = 'it'
# The set of all layers in this model, in order of processing.
ALL_LAYERS = (LAYER_RETINA, LAYER_S1, LAYER_C1, LAYER_S2, LAYER_C2, LAYER_IT)
ALL_BUILDERS = (Viz2Model.BuildRetinaFromImage, Viz2Model.BuildS1FromRetina,
    Viz2Model.BuildC1FromS1, Viz2Model.BuildS2FromC1, Viz2Model.BuildC2FromS2,
    Viz2Model.BuildItFromC2)

def Whiten(data):
  data -= data.mean(0)
  norms = np.sqrt((data**2).sum(0))
  norms[ norms < 1 ] = 1
  data /= norms
  return data

def CheckPrototypes(prototypes):
  assert prototypes != None
  if len(prototypes.shape) == 3:
    prototypes = prototypes.reshape((1,) + prototypes.shape)
  assert np.allclose(np.array(map(np.linalg.norm, prototypes)), 1), \
      "Internal error: S2 prototypes are not normalized"
  assert not np.isnan(prototypes).any(), \
      "Internal error: found NaN in imprinted prototype."

#### OPTION HANDLING ###########

ALL_OPTIONS = [
  ('retina_bias', "Term added to standard deviation of local window"),
  ('retina_enabled', "Indicates whether the retinal layer is used"),
  ('retina_kwidth', "Spatial width of input neighborhood for retinal units"),

  ('s1_beta', "Beta parameter of RBF for S1 cells"),
  ('s1_bias', "Term added to the norm of the input vector"),
  ('s1_kwidth', "Spatial width of input neighborhood for S1 units"),
  ('s1_num_orientations', "Number of different S1 Gabor orientations"),
  ('s1_num_phases', """Number of different phases for S1 Gabors. Using two
                    phases corresponds to find a light bar on a dark
                    background and vice versa"""),
  ('s1_scaling', """Subsampling factor (e.g., setting s1_scaling=2 will result
                 in an output array that is 1/2 the width and half the
                 height of the input array)"""),
  ('s1_shift_orientations', "Rotate Gabors by a small positive angle"),

  ('c1_kwidth', "Spatial width of input neighborhood for C1 units"),
  ('c1_scaling', "Subsampling factor"),
  ('c1_whiten', "Whether to normalize the total energy at each C1 location"),

  ('s2_beta', "Beta parameter of RBF for S1 cells"),
  ('s2_bias', "Additive term combined with input window norm"),
  ('s2_kwidth', "Spatial width of input neighborhood for S2 units"),
  ('s2_scaling', "Subsampling factor"),

  ('c2_kwidth', "Spatial width of input neighborhood for C2 units"),
  ('c2_scaling', "Subsampling factor"),

  ('num_scales', "Number of different scale bands"),
]

class Viz2Params(object):

  def __init__(self, **args):
    self._params = MakeDefaultParamDict()
    for name, value in args.items():
      self[name] = value

  def __repr__(self):
    return str(self._params)

  def __eq__(self, params):
    return params != None and self._params == params._params

  def __getitem__(self, name):
    if name not in self._params:
      raise ValueError("Unknown Viz2 param: %s" % name)
    return self._params[name]

  def __setitem__(self, name, value):
    assert name in self._params
    self._params[name] = value

  def __getstate__(self):
    return self._params

  def __setstate__(self, state):
    self._params = MakeDefaultParamDict()
    for name, value in state.items():
      self[name] = value

  def LoadFromFile(self, fname):
    """Loads option data either from a python file (if fname ends in ".py"), or
    a pickled dictionary.
      fname -- name of file from which to load options"""
    for name, value in util.LoadByFileName(fname).items():
      self[name] = value

def MakeDefaultParamDict():
  """Create a default set of parameters."""
  return dict(
    retina_bias = 1.0,
    retina_enabled = True,
    retina_kwidth = 15,

    s1_bias = 1.0,
    s1_beta = 1.0,
    s1_kwidth = 11,
    s1_num_orientations = 8,
    s1_num_phases = 2,
    s1_scaling = 2,
    s1_shift_orientations = True,
    s1_sparsify = False,

    c1_kwidth = 5,
    c1_scaling = 2,
    c1_sparsify = False,
    c1_whiten = True,

    s2_beta = 5.0,
    s2_bias = 1.0,
    s2_kwidth = 7,
    s2_scaling = 2,

    c2_kwidth = 3,
    c2_scaling = 2,

    num_scales = 4,
  )

#### COMMAND LINE INTERFACE ###########

from glimpse.backends.cython_backend import CythonBackend
from glimpse.backends.scipy_backend import ScipyBackend
import Image
import os
import sys

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
        PrintParamHelp()
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
    ifname = args[0]
    img = Image.open(ifname)
    img = util.ImageToInputArray(img)
    model = MakeModel(backend, params_fname)
    protos, locations = model.ImprintPrototypes(img, num_prototypes)
    if stream == True:
      for p in protos:
        util.Store(p, sys.stdout)
    else:
      util.Store(np.array(protos), sys.stdout)
    if print_locations == True:
      for loc in locations:
        print >>sys.stderr, ifname + " " + " ".join(map(str, loc))
  except util.UsageException, e:
    if e.msg:
      print >>sys.stderr, e.msg
    util.Usage("[options] IMAGE\n"
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

# Identifiers for objects that can be stored
STORAGE_ELEMENTS = set(['options', 'image', 's1-kernels', 's2-kernels'] +
    [ "%s-activity" % x for x in ALL_LAYERS ])

def Transform(args):
  backend = "cython"
  store_list = "none"
  output_layer = LAYER_IT
  try:
    params_fname = None
    proto_fname = None
    rdir = None
    print_it = False
    protos = None
    opts, args = util.GetOptions("b:Hil:o:p:r:s:", args = args)
    for opt, arg in opts:
      if opt == '-b':
        backend = arg
      elif opt == '-i':
        print_it = True
      elif opt == '-H':
        PrintParamHelp()
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
    ifname = args[0]
    img = Image.open(ifname)
    img = util.ImageToInputArray(img)
    model = MakeModel(backend, params_fname, s2_kernels = protos)
    results = model.BuildLayers(img, LAYER_RETINA, output_layer)
    if rdir != None:
      results_for_output = dict([ ("%s-activity" % k, v) for k, v in results.items() ])
      results_for_output['options'] = model.params
      results_for_output['s1-kernels'] = model.s1_kernels
      results_for_output['s2-kernels'] = model.s2_kernels
      for name in set(results_for_output.keys()).intersection(store_list):
        fname = os.path.join(rdir, name)
        util.Store(results_for_output[name], fname)
    if print_it:
      util.Store(results[LAYER_IT], sys.stdout)
  except util.UsageException, e:
    if e.msg:
      print >>sys.stderr, e.msg
    util.Usage("[options] IMAGE\n"
        "  -b STR   Set backend type (one of cython, scipy."
        " default: %s)\n" % backend + \
        "  -i       Write IT data to stdout\n"
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

def Main(args):
  try:
    if len(args) < 1:
      raise util.UsageException()
    cmd = args[0].lower()
    if cmd == "imprint":
      Imprint(args[1:])
    elif cmd == "transform":
      Transform(args[1:])
    else:
      raise util.UsageException("Unknown command: %s" % cmd)
  except util.UsageException, e:
    util.Usage("[imprint|transform]", e)
