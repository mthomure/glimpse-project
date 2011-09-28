# Create a 2-part, HMAX-like hierarchy of S+C layers.
# This module implements the "Viz2" model used for the GCNC 2011 experiments.


# Why do larger scales have more energy than smaller scales (on B_ani30)? Is
# this consistent across images? Do the images just have more content at lower
# scales?


from glimpse import core, util
from glimpse.core import activation_dtype
import numpy as np
import Image
import random
import itertools
import os
import sys

from glimpse.backends.scipy_backend import ScipyBackend
from glimpse.backends.cython_backend import CythonBackend

TIMER = False


def PrototypeSampler(c1s, num_prototypes, kwidth, scale_norm):
  """
  c1 - list of (3-D) C1 activity maps, one map per scale.
  RETURNS: iterator over prototype arrays. Note that returned prototypes should
           be copied, as the returned array may be reused.
  """
  assert (np.array(len(c1s[0].shape)) == 3).all()
  num_scales = len(c1s)
  num_orientations = len(c1s[0])
  #~ proto = np.empty([num_orientations, kwidth, kwidth], activation_dtype)
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

  def __init__(self, backend, params, logger = None):
    self.backend = backend
    self.params = params
    self.logger = logger

  def _Log(self, key = None, value = None, **args):
    if self.logger == None:
      return
    if key != None:
      args[key] = value
    self.logger(**args)

  def _LogScale(self, scale_idx, key = None, value = None, **args):
    if self.logger == None:
      return
    if key != None:
      args[key] = value
    self.logger(**dict([ ("%s-%d" % (key, scale_idx), value)
        for key, value in args.items() ]))

  def BuildThroughC1(self, img):
    num_scales = self.params['num_scales']
    s1_kwidth = self.params['s1_kwidth']
    s1_num_orientations = self.params['s1_num_orientations']
    s1_num_phases = self.params['s1_num_phases']
    mks = core.MakeMultiScaleGaborKernels(kwidth = s1_kwidth,
        num_scales = num_scales, num_orientations = s1_num_orientations,
        num_phases = s1_num_phases, shift_orientations = True,
        scale_norm = True)
    # Reshape kernel array to be 4-D: scale, index, 1, y, x
    mks_ = mks.reshape((num_scales, -1, 1, s1_kwidth, s1_kwidth))
    img = core.ImageToInputArray(img)
    retina = self.backend.ContrastEnhance(img,
        kwidth = self.params['retina_kwidth'],
        bias = self.params['retina_bias'])
    self._Log("s1-kernels", mks)
    self._Log(image = img, retina = retina)
    retina_ = retina.reshape((1,) + retina.shape)
    scale_data = []
    for scale_idx, ks in zip(range(len(mks_)), mks_):
      s1_ = self.backend.NormRbf(retina_, ks, bias = self.params['s1_bias'],
          beta = self.params['s1_beta'], scaling = self.params['s1_scaling'])
      s1 = s1_.reshape((s1_num_orientations, s1_num_phases) + s1_.shape[-2:])
      # Pool over phase
      s1 = s1.max(1)
      c1 = self.backend.LocalMax(s1, kwidth = self.params['c1_kwidth'],
          scaling = self.params['c1_scaling'])

      # DEBUG: temporarily removed
      #~ if self.params['c1_whiten']:
        #~ Viz2Model.Whiten(c1)

      scale_data.append((s1, c1))

    # DEBUG: viz2 used pann's whitening method, which normalized across scales
    if self.params['c1_whiten']:
      s1s, c1s = zip(*scale_data)
      c1s = np.array(c1s, activation_dtype)
      c1_shape = c1s.shape
      c1s = c1s.reshape((-1,) + c1s.shape[-2:])
      Viz2Model.Whiten(c1s)
      c1s = c1s.reshape(c1_shape)
      scale_data = zip(s1s, c1s)

    for scale_idx in range(num_scales):
      s1, c1 = scale_data[scale_idx]
      self._LogScale(scale_idx, "s1-activity", s1)
      self._LogScale(scale_idx, "c1-activity", c1)
    return scale_data

  @classmethod
  def Whiten(cls, data):
    data -= data.mean(0)
    norms = np.sqrt((data**2).sum(0))
    norms[ norms < 1 ] = 1
    data /= norms
    return data

  @classmethod
  def CheckPrototypes(cls, prototypes):
    if len(prototypes.shape) == 3:
      prototypes = prototypes.reshape((1,) + prototypes.shape)
    assert np.allclose(np.array(map(np.linalg.norm, prototypes)), 1), \
        "Internal error: S2 prototypes are not normalized"
    assert not np.isnan(prototypes).any(), \
        "Internal error: found NaN in imprinted prototype."

  def BuildItFromC1(self, c1s, s2_prototypes):
    """Compute location- and scale-invariant features.
    c1s - list of (3-D) C1 maps, one map per scale
    """
    Viz2Model.CheckPrototypes(s2_prototypes)
    c2s = []
    for scale_idx, c1 in zip(range(len(c1s)), c1s):
      s2 = self.backend.NormRbf(c1, s2_prototypes,
          bias = self.params['s2_bias'], beta = self.params['s2_beta'],
          scaling = self.params['s2_scaling'])
      c2 = self.backend.GlobalMax(s2)
      c2s.append(c2)
      self._LogScale(scale_idx, "s2-activity", s2)
      self._LogScale(scale_idx, "c2-activity", c2)
    it = np.array(c2s).max(0)
    self._Log("it-activity", it)
    return it

  def ImprintPrototypes(self, img, num_prototypes):
    """Compute C1 activity maps and sample patches from random locations.
    RETURNS list of prototypes, and list of corresponding locations.
    NOTE: prototype values should be copied, as array may be reused
    """
    c1s = [ c1 for s1, c1 in self.BuildThroughC1(img) ]
    proto_it = PrototypeSampler(c1s, num_prototypes,
        kwidth = self.params['s2_kwidth'], scale_norm = True)
    return zip(*list(proto_it))


#### OPTION HANDLING ###########

class Viz2Params(object):

  PARAMS = [
    ('retina_enabled', "Indicates whether the retinal layer is used"),
    ('retina_bias', "Term added to standard deviation of local window"),
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
    ('s1_sparsify', "Supress activation for non-maximal edge orientations at S1"),

    ('c1_kwidth', "Spatial width of input neighborhood for C1 units"),
    ('c1_scaling', "Subsampling factor"),
    ('c1_sparsify', "Supress activation for non-maximal edge orientations at C1"),
    ('c1_whiten', "Whether to normalize the total energy at each C1 location"),

    ('s2_beta', "Beta parameter of RBF for S1 cells"),
    ('s2_bias', "Additive term combined with input window norm"),
    ('s2_kwidth', "Spatial width of input neighborhood for S2 units"),
    ('s2_scaling', "Subsampling factor"),

    ('c2_kwidth', "Spatial width of input neighborhood for C2 units"),
    ('c2_scaling', "Subsampling factor"),

    ('num_scales', "Number of different scale bands"),
    ('scale_factor', "Image downsampling factor between scale bands"),
    ('sse_enabled', "Enable use of SSE intrinsics, when available"),
  ]

  def __init__(self, **args):
    self._params = dict(
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
      scale_factor = 2**(1./2),
      sse_enabled = core.GetUseSSE(),
    )
    for name, value in args.items():
      self[name] = value

  def __getitem__(self, name):
    assert name in self._params
    return self._params[name]

  def __setitem__(self, name, value):
    assert name in self._params
    self._params[name] = value

  def __getstate__(self):
    return self._params

  def __setstate__(self, state):
    for name, value in args.items():
      self[name] = value

  def ApplyGlobalParams(self):
    """Apply all global option entries."""
    core.SetUseSSE(self['sse_enabled'])

  def LoadFromFile(self, fname):
    """Loads option data either from a python file (if fname ends in ".py"), or
    a pickled dictionary.
      fname -- name of file from which to load options"""
    for name, value in util.LoadByFileName(fname).items():
      self[name] = value


###### COMMAND LINE INTERFACE ############

def MakeDirLogger(dname):
  def logger(**args):
    for key, value in args.items():
      fname = "%s.dat" % os.path.join(dname, key)
      util.Store(value, fname)
  return logger

def MakeModel(backend = "cython", params_fname = None):
  mod_name = "%sBackend" % backend.capitalize()
  backend = eval("%s()" % mod_name)
  params = Viz2Params()
  if params_fname != None:
    params.LoadFromFile(params_fname)
  params.ApplyGlobalParams()
  model = Viz2Model(backend, params)

  #~ print "Model backend: %s" % model.backend

  return model

def main():
  backend = "cython"
  params_fname = None
  proto_fname = None
  rdir = None
  opts, args = util.GetOptions("b:o:p:r:", [])
  for opt, arg in opts:
    if opt == '-b':
      backend = arg
    elif opt == '-o':
      params_fname = arg
    elif opt == '-p':
      proto_fname = arg
    elif opt == '-r':
      rdir = arg
  if len(args) < 2:
    sys.exit("usage: %s CMD IMAGE" % sys.argv[0])
  cmd, ifname = args[:2]
  args = args[2:]
  model = MakeModel(backend, params_fname)
  if rdir != None:
    model.logger = MakeDirLogger(rdir)
    model.logger(params = model.params)
  img = Image.open(ifname)
  if cmd.upper() == "IMPRINT":
    if len(args) < 1:
      sys.exit("usage: %s IMPRINT IMAGE NUM_PROTOS > PROTOS" % sys.argv[0])
    num_prototypes = int(args[0])
    with util.Timer('ImprintPrototypes', TIMER):
      protos, locations = model.ImprintPrototypes(img, num_prototypes)
      for loc in locations:
        print >>sys.stderr, ifname + " " + " ".join(map(str, loc))
      util.Store(np.array(protos), sys.stdout)
  elif cmd.upper() == "TRANSFORM":
    if proto_fname == None:
      sys.exit("Missing S2 prototypes")
    protos = util.Load(proto_fname)
    c1s = [ c1 for s1, c1 in model.BuildThroughC1(img) ]
    it = model.BuildItFromC1(c1s, protos)
    #~ util.Store(it, sys.stdout)
  else:
    sys.exit("bad operation '%s': should be one of IMPRINT or TRANSFORM" % cmd)

if __name__ == "__main__":
  main()
