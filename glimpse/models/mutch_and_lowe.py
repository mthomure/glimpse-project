# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# Create a 2-part, HMAX-like hierarchy of S+C layers.
# This module implements the scale-pyramid approach used by Mutch & Lowe (2008).

from glimpse.util import kernel
from scipy.ndimage.interpolation import zoom
from viz2 import PrototypeSampler, Whiten, CheckPrototypes, \
    ImageLayerFromInputArray
from viz2 import LAYER_IMAGE, LAYER_C1, ALL_LAYERS
import numpy as np

class Model(object):

  def __init__(self, backend, params, s1_kernels = None, s2_kernels = None):
    self.backend = backend
    self.params = params
    if s1_kernels == None:
      s1_kernels = kernel.MakeGaborKernels(
          kwidth = params['s1_kwidth'],
          num_orientations = params['s1_num_orientations'],
          num_phases = params['s1_num_phases'], shift_orientations = True,
          scale_norm = True)
    self.s1_kernels = s1_kernels
    self.s2_kernels = s2_kernels

  def BuildImageFromInput(self, input_):
    return ImageLayerFromInputArray(input_, self.backend)

  def BuildRetinaFromImage(self, img):
    if not self.params['retina_enabled']:
      return img
    retina = self.backend.ContrastEnhance(img,
        kwidth = self.params['retina_kwidth'],
        bias = self.params['retina_bias'],
        scaling = 1)
    return retina

  def BuildS1FromRetina(self, retina):
    # Create scale pyramid of retinal map
    num_scales = self.params['num_scales']
    factor = self.params['scale_factor']
    rs = [ zoom(retina, 1 / factor ** scale) for scale in range(num_scales) ]
    s1_kwidth = self.params['s1_kwidth']
    s1_num_orientations = self.params['s1_num_orientations']
    s1_num_phases = self.params['s1_num_phases']
    # Reshape kernel array to be 3-D: index, 1, y, x
    s1_kernels = self.s1_kernels.reshape((-1, 1, s1_kwidth, s1_kwidth))
    s1s = []
    for scale in range(num_scales):
      # Reshape retinas to be 3D array
      r = rs[scale]
      retina_ = r.reshape((1,) + r.shape)
      s1_ = self.backend.NormRbf(retina_, s1_kernels,
          bias = self.params['s1_bias'], beta = self.params['s1_beta'],
          scaling = self.params['s1_scaling'])
      # Reshape S1 to be 4D array
      s1 = s1_.reshape((s1_num_orientations, s1_num_phases) + s1_.shape[-2:])
      s1 = s1.max(1)
      s1s.append(s1)
    return s1s

  def BuildC1FromS1(self, s1s):
    num_scales = self.params['num_scales']
    c1s = [ self.backend.LocalMax(s1, kwidth = self.params['c1_kwidth'],
        scaling = self.params['c1_scaling']) for s1 in s1s ]
    # Whiten over orientation only
    if self.params['c1_whiten']:
      for c1 in c1s:
        Whiten(c1)
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
    results = self.BuildLayers(img, LAYER_IMAGE, LAYER_C1)
    c1s = results[LAYER_C1]
    proto_it = PrototypeSampler(c1s, num_prototypes,
        kwidth = self.params['s2_kwidth'], scale_norm = True)
    protos = list(proto_it)
    return zip(*protos)

ALL_BUILDERS = (Model.BuildImageFromInput, Model.BuildRetinaFromImage,
    Model.BuildS1FromRetina, Model.BuildC1FromS1,
    Model.BuildS2FromC1, Model.BuildC2FromS2, Model.BuildItFromC2)

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
  ('scale_factor', "Image downsampling factor between scale bands"),
]

class Params(object):

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
      raise ValueError("Unknown parameter: %s" % name)
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

    c1_kwidth = 5,
    c1_scaling = 2,
    c1_whiten = True,

    s2_beta = 5.0,
    s2_bias = 1.0,
    s2_kwidth = 7,
    s2_scaling = 2,

    c2_kwidth = 3,
    c2_scaling = 2,

    num_scales = 4,
    scale_factor = 2**0.5,
  )
