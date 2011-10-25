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
import Image
import numpy as np
import random

def PrototypeSampler(c1s, num_prototypes, kwidth, scale_norm):
  """
  c1 - list of (3-D) C1 activity maps, one map per scale.
  RETURNS: iterator over prototype arrays and corresponding locations (i.e.,
           iterator elements are 2-tuples). Prototype location gives top-left
           corner of C1 region.
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

def ImageLayerFromInputArray(input_, backend):
  """Create the initial image layer from some input.
  input_ -- Image or (2-D) array of input data. If array, values should lie in
            the range [0, 1].
  RETURNS (2-D) array containing image layer data
  """
  if isinstance(input_, Image.Image):
    input_ = input_.convert('L')
    input_ = util.ImageToArray(input_, transpose = True)
    input_ = input_.astype(np.float)
    # Map from [0, 255] to [0, 1]
    input_ /= 255
  return backend.PrepareArray(input_)

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
    else:
      # S1 kernels should have dimensions [scale, orientation, phase, y, x].
      expected_shape = tuple(params[k] for k in ('num_scales',
          's1_num_orientations', 's1_num_phases', 's1_kwidth', 's1_kwidth'))
      assert s1_kernels.shape == expected_shape, \
          "S1 kernels have wrong shape: expected %s but got %s" % \
          (expected_shape, s1_kernels.shape)
      s1_kernels = backend.PrepareArray(s1_kernels)
    if s2_kernels != None:
      # S2 kernels should have dimensions [proto_idx, orientation, y, x].
      ntheta = params['s1_num_orientations']
      kwidth = params['s1_kwidth']
      assert s2_kernels.ndim == 4 and s2_kernels.shape[0] > 0 \
          and s2_kernels.shape[1:] == (ntheta, kwidth, kwidth), \
          "S2 kernels have wrong shape: expected (*, %d, %d, %d) but got %s" % \
          (ntheta, kwidth, kwidth, s2_kernels.shape)
      s2_kernels = backend.PrepareArray(s2_kernels)
    self.s1_kernels = s1_kernels
    self.s2_kernels = s2_kernels

  def BuildImageFromInput(self, input_):
    """Create the initial image layer from some input.
    input_ -- Image or (2-D) array of input data. If array, values should lie in
              the range [0, 1].
    RETURNS (2-D) array containing image layer data
    """
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
    """Apply S1 processing to some existing retinal layer data.
    retina -- (2-D array) result of retinal layer processing
    RETURNS list of (4-D) S1 activity arrays, with one array per scale
    """
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
      # Pool over phase.
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
    """Process the input data, building up to the given output layer.
    data -- input array
    input_layer -- layer ID of input array
    output_layer -- layer ID of output array
    """
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

  def ImprintPrototypes(self, input_, num_prototypes):
    """Compute C1 activity maps and sample patches from random locations.
    RETURNS list of prototypes, and list of corresponding locations.
    """
    results = self.BuildLayers(input_, LAYER_IMAGE, LAYER_C1)
    c1s = results[LAYER_C1]
    proto_it = PrototypeSampler(c1s, num_prototypes,
        kwidth = self.params['s2_kwidth'], scale_norm = True)
    protos = list(proto_it)
    return zip(*protos)

# Identifiers for layers that can be computed
LAYER_IMAGE = 'i'
LAYER_RETINA = 'r'
LAYER_S1 = 's1'
LAYER_C1 = 'c1'
LAYER_S2 = 's2'
LAYER_C2 = 'c2'
LAYER_IT = 'it'
# The set of all layers in this model, in order of processing.
ALL_LAYERS = (LAYER_IMAGE, LAYER_RETINA, LAYER_S1, LAYER_C1, LAYER_S2, LAYER_C2,
    LAYER_IT)
ALL_BUILDERS = (Viz2Model.BuildImageFromInput, Viz2Model.BuildRetinaFromImage,
    Viz2Model.BuildS1FromRetina, Viz2Model.BuildC1FromS1,
    Viz2Model.BuildS2FromC1, Viz2Model.BuildC2FromS2, Viz2Model.BuildItFromC2)

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

#### COORDINATE MAPPING ########

class CoordinateMapper(object):
  """Given an offset expressed in the coordinates of one layer, these methods
  convert that value to the equivalent offset in the coordinates of a lower
  layer. Note that these methods assume that all kernels are square."""

  def __init__(self, params, center = True):
    self.params = params
    self.center = center

  def GetMappingFunction(self, activity_layer, background_layer):
    """Lookup the function that maps activity from one layer to another, where
    layers are identified by strings (e.g., "s1", or "image")."""
    def clean(layer):
      if layer == LAYER_IMAGE:
        layer = 'image'
      if layer == LAYER_RETINA:
        layer = 'retina'
      return layer.title()
    try:
      return getattr(self, "Map%sTo%s" % (clean(activity_layer),
          clean(background_layer)))
    except AttributeError:
      raise ValueError("No function found to map from %s to %s" %
          (activity_layer, background_layer))

  def MapRetinaToImage(self, x):
    if self.params['retina_enabled'] and self.center:
      r_kw = self.params['retina_kwidth']
      x += r_kw / 2
    return int(x)

  def MapS1ToRetina(self, x):
    s1_s = self.params['s1_scaling']
    x = x * s1_s
    if self.center:
      s1_kw = self.params['s1_kwidth']
      x += s1_kw / 2
    return int(x)

  def MapS1ToImage(self, x):
    return self.MapRetinaToImage(self.MapS1ToRetina(x))

  def MapC1ToS1(self, x):
    c1_s = self.params['c1_scaling']
    x = x * c1_s
    if self.center:
      c1_kw = self.params['c1_kwidth']
      x += c1_kw / 2
    return int(x)

  def MapC1ToRetina(self, x):
    return self.MapS1ToRetina(self.MapC1ToS1(x))

  def MapC1ToImage(self, x):
    return self.MapRetinaToImage(self.MapC1ToRetina(x))

  def MapS2ToC1(self, x):
    s2_s = self.params['s2_scaling']
    x = x * s2_s
    if self.center:
      s2_kw = self.params['s2_kwidth']
      x += s2_kw / 2
    return int(x)

  def MapS2ToImage(self, x):
    return self.MapC1ToImage(self.MapS2ToC1(x))

  def MapC2ToS2(self, x):
    c2_s = self.params['c2_scaling']
    x = x * c2_s
    if self.center:
      c2_kw = self.params['c2_kwidth']
      x += c2_kw / 2
    return int(x)

  def MapC2ToImage(self, x):
    return self.MapS2ToImage(self.MapC2ToS2(x))

class LayerSizeMapper(object):
  """These methods compute the image size required to support a model layer
  (e.g., C2) with at least the given spatial extent (e.g., 3x3 units)."""

  def __init__(self, backend, params):
    self.backend = backend
    self.params = params

  def GetMappingFunction(self, activity_layer, background_layer):
    """Lookup the function that maps activity from one layer to another, where
    layers are identified by strings (e.g., "s1", or "image")."""
    def clean(layer):
      if layer == LAYER_IMAGE:
        layer = 'image'
      if layer == LAYER_RETINA:
        layer = 'retina'
      return layer.title()
    try:
      return getattr(self, "Map%sTo%s" % (clean(activity_layer),
          clean(background_layer)))
    except AttributeError:
      raise ValueError("No function found to map from %s to %s" %
          (activity_layer, background_layer))

  def MapRetinaToImage(self, height, width):
    kw = self.params['retina_kwidth']
    return self.backend.InputMapShapeForOutput(kw, kw,
        1, # scaling
        height, width)

  def MapS1ToRetina(self, height, width):
    kw = self.params['s1_kwidth']
    scaling = self.params['s1_scaling']
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapS1ToImage(self, height, width):
    return self.MapRetinaToImage(*self.MapS1ToRetina(height, width))

  def MapC1ToS1(self, height, width):
    kw = self.params['c1_kwidth']
    scaling = self.params['c1_scaling']
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapC1ToImage(self, height, width):
    return self.MapS1ToImage(*self.MapC1ToS1(height, width))

  def MapS2ToC1(self, height, width):
    kw = self.params['s2_kwidth']
    scaling = self.params['s2_scaling']
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapS2ToImage(self, height, width):
    return self.MapC1ToImage(*self.MapS2ToC1(height, width))

  def MapC2ToS2(self, height, width):
    kw = self.params['c2_kwidth']
    scaling = self.params['c2_scaling']
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapC2ToImage(self, height, width):
    return self.MapS2ToImage(*self.MapC2ToS2(height, width))

class RegionMapper(object):
  """The methods of this object re-express a slice --- given in coordinates
  of one layer --- in terms of coordinates of some lower-level layer. Note
  that these methods assume that all kernels are square."""

  def __init__(self, params):
    self.params = params
    self.cmapper = CoordinateMapper(params, center = False)

  def GetMappingFunction(self, activity_layer, background_layer):
    """Lookup the function that maps activity from one layer to another, where
    layers are identified by strings (e.g., "s1", or "image").
    activity_layer -- identifier of foreground layer
    background_layer -- identifier of background layer
    """
    # NOTE: uses the fact that layer identifiers are short strings
    def clean(layer):
      if layer == LAYER_IMAGE:
        layer = 'image'
      if layer == LAYER_RETINA:
        layer = 'retina'
      return layer.title()
    try:
      return getattr(self, "Map%sTo%s" % (clean(activity_layer),
          clean(background_layer)))
    except AttributeError:
      raise ValueError("No function found to map from %s to %s" %
          (activity_layer, background_layer))

  def MapRetinaToImage(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapRetinaToImage
    if self.params['retina_enabled']:
      offset = self.params['retina_kwidth']
    else:
      offset = 1  # retina width is 1 unit
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapS1ToRetina(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapS1ToRetina
    offset = self.params['s1_kwidth']
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapS1ToImage(self, x):
    return self.MapRetinaToImage(self.MapS1ToRetina(x))

  def MapC1ToS1(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapC1ToS1
    offset = self.params['c1_kwidth']
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapC1ToRetina(self, x):
    return self.MapS1ToRetina(self.MapC1ToS1(x))

  def MapC1ToImage(self, x):
    """Map C1 region to image coordinates.
    x -- (slice) region in C1 coordinates
    """
    return self.MapRetinaToImage(self.MapC1ToRetina(x))

  def MapS2ToC1(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapS2ToC1
    offset = self.params['s2_kwidth']
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapS2ToImage(self, x):
    return self.MapC1ToImage(self.MapS2ToC1(x))

  def MapC2ToS2(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapC2ToS2
    offset = self.params['c2_kwidth']
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapC2ToImage(self, x):
    return self.MapS2ToImage(self.MapC2ToS2(x))

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
    a pickled Viz2Params file.
      fname -- name of file from which to load options"""
    values = util.LoadByFileName(fname)
    if isinstance(values, Viz2Params):
      values = values._params
    elif not isinstance(values, dict):
      raise ValueError("Unknown data in file %s" % fname)
    for name, value in values.items():
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
    s2_bias = 0.1,  # Configured to match distribution of C1 norm under
                    # whitening.
    s2_kwidth = 7,
    s2_scaling = 2,

    c2_kwidth = 3,
    c2_scaling = 2,

    num_scales = 4,
  )
