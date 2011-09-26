
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Functions for transforming an image to a set of features using a hierarchical,
# HMAX-like model.
#

from glimpse.core import c_src
from glimpse.core.c_src import activation_dtype
from glimpse.core import misc
from glimpse import util
from glimpse.util import array
from glimpse.util import bitset
import Image
import math
import numpy as np
import os
import random
import sys

# Identifiers for layers that can be computed
LAYER_RETINA = 'r'
LAYER_S1 = 's1'
LAYER_C1 = 'c1'
LAYER_S2 = 's2'
LAYER_C2 = 'c2'
LAYER_IT = 'it'
LAYERS = (LAYER_RETINA, LAYER_S1, LAYER_C1, LAYER_S2, LAYER_C2, LAYER_IT)

# Identifiers for objects that can be stored
STORAGE_ELEMENTS = set(('options', 'image', 'r-activity', 's1-kernels',
    's1-activity', 'c1-activity', 'c1-coords', 's2-activity', 'c2-activity',
    'c2-coords', 'global-c2-activity', 'global-c2-coords', 'it-activity',
    'it-coords', 'feature-vector',
    ))

class CoordinateMapper(object):
  """Given an offset expressed in the coordinates of one layer, these methods
  convert that value to the equivalent offset in the coordinates of a lower
  layer. Note that these methods assume that all kernels are square."""

  def __init__(self, options = None, center = True):
    if options == None:
      options = MakeDefaultOptions()
    self.options = options
    self.center = center
    self._maps = [
        'MapRetinaToImage',
        'MapS1ToRetina',
        'MapS1ToImage',
        'MapC1ToS1',
        'MapC1ToRetina',
        'MapC1ToImage',
        'MapS2ToC1',
        'MapS2ToImage',
        'MapC2ToS2',
        'MapC2ToImage',
    ]

  def GetMappingFunction(self, activity_layer, background_layer):
    """Lookup the function that maps activity from one layer to another, where
    layers are identified by strings (e.g., "s1", or "image")."""
    def clean(name):
      if name == 'i':
        name = 'image'
      if name == 'r':
        name = 'retina'
      return name.title()
    try:
      return getattr(self, "Map%sTo%s" % (clean(activity_layer),
          clean(background_layer)))
    except AttributeError:
      raise ValueError("No function found to map from %s to %s" %
          (activity_layer, background_layer))

  def MapRetinaToImage(self, x):
    if self.options['retina_enabled'] and self.center:
      r_kw = self.options['retina_kwidth']
      x += r_kw / 2
    return int(x)

  def MapS1ToRetina(self, x):
    s1_s = self.options['s1_scaling']
    x = x * s1_s
    if self.center:
      s1_kw = self.options['s1_kwidth']
      x += s1_kw / 2
    return int(x)

  def MapS1ToImage(self, x):
    return self.MapRetinaToImage(self.MapS1ToRetina(x))

  def MapC1ToS1(self, x):
    c1_s = self.options['c1_scaling']
    x = x * c1_s
    if self.center:
      c1_kw = self.options['c1_kwidth']
      x += c1_kw / 2
    return int(x)

  def MapC1ToRetina(self, x):
    return self.MapS1ToRetina(self.MapC1ToS1(x))

  def MapC1ToImage(self, x):
    return self.MapRetinaToImage(self.MapC1ToRetina(x))

  def MapS2ToC1(self, x):
    s2_s = self.options['s2_scaling']
    x = x * s2_s
    if self.center:
      s2_kw = self.options['s2_kwidth']
      x += s2_kw / 2
    return int(x)

  def MapS2ToImage(self, x):
    return self.MapC1ToImage(self.MapS2ToC1(x))

  def MapC2ToS2(self, x):
    c2_s = self.options['c2_scaling']
    x = x * c2_s
    if self.center:
      c2_kw = self.options['c2_kwidth']
      x += c2_kw / 2
    return int(x)

  def MapC2ToImage(self, x):
    return self.MapS2ToImage(self.MapC2ToS2(x))


class LayerSizeMapper(object):
  """These methods compute the image size required to support a model layer
  (e.g., C2) with at least the given spatial extent (e.g., 3x3 units)."""

  def __init__(self, options = None):
    if options == None:
      options = MakeDefaultOptions()
    self.options = options

  def GetMappingFunction(self, activity_layer, background_layer):
    """Lookup the function that maps activity from one layer to another, where
    layers are identified by strings (e.g., "s1", or "image")."""
    def clean(name):
      if name == 'i':
        name = 'image'
      if name == 'r':
        name = 'retina'
      return name.title()
    try:
      return getattr(self, "Map%sTo%s" % (clean(activity_layer),
          clean(background_layer)))
    except AttributeError:
      raise ValueError("No function found to map from %s to %s" %
          (activity_layer, background_layer))

  def MapRetinaToImage(self, height, width):
    kw = self.options['retina_kwidth']
    sse = self.options['sse_enabled']
    return c_src.InputDimensionsForOutput(kw, kw, 1, height, width, sse)

  def MapS1ToRetina(self, height, width):
    kw = self.options['s1_kwidth']
    scaling = self.options['s1_scaling']
    sse = self.options['sse_enabled'] and scaling == 1
    return c_src.InputDimensionsForOutput(kw, kw, scaling, height, width, sse)

  def MapS1ToImage(self, height, width):
    return self.MapRetinaToImage(*self.MapS1ToRetina(height, width))

  def MapC1ToS1(self, height, width):
    kw = self.options['c1_kwidth']
    sse = False
    return c_src.InputDimensionsForOutput(kw, kw, self.options['c1_scaling'],
        height, width, sse)

  def MapC1ToImage(self, height, width):
    return self.MapS1ToImage(*self.MapC1ToS1(height, width))

  def MapS2ToC1(self, height, width):
    kw = self.options['s2_kwidth']
    scaling = self.options['s2_scaling']
    sse = self.options['sse_enabled'] and scaling == 1
    return c_src.InputDimensionsForOutput(kw, kw, scaling, height, width, sse)

  def MapS2ToImage(self, height, width):
    return self.MapC1ToImage(*self.MapS2ToC1(height, width))

  def MapC2ToS2(self, height, width):
    kw = self.options['c2_kwidth']
    sse = False
    return c_src.InputDimensionsForOutput(kw, kw, self.options['c2_scaling'],
        height, width, sse)

  def MapC2ToImage(self, height, width):
    return self.MapS2ToImage(*self.MapC2ToS2(height, width))


class RegionMapper(object):
  """The methods of this object re-express a slice --- given in coordinates
  of one layer --- in terms of coordinates of some lower-level layer. Note
  that these methods assume that all kernels are square."""

  def __init__(self, options = None):
    if options == None:
      options = MakeDefaultOptions()
    self.options = options
    self.cmapper = CoordinateMapper(options, center = False)

  def GetMappingFunction(self, activity_layer, background_layer):
    """Lookup the function that maps activity from one layer to another, where
    layers are identified by strings (e.g., "s1", or "image")."""
    def clean(name):
      if name == 'i':
        name = 'image'
      if name == 'r':
        name = 'retina'
      return name.title()
    try:
      return getattr(self, "Map%sTo%s" % (clean(activity_layer),
          clean(background_layer)))
    except AttributeError:
      raise ValueError("No function found to map from %s to %s" %
          (activity_layer, background_layer))

  def MapRetinaToImage(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapRetinaToImage
    if self.options['retina_enabled']:
      offset = self.options['retina_kwidth']
    else:
      offset = 1  # retina width is 1 unit
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapS1ToRetina(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapS1ToRetina
    offset = self.options['s1_kwidth']
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapS1ToImage(self, x):
    return self.MapRetinaToImage(self.MapS1ToRetina(x))

  def MapC1ToS1(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapC1ToS1
    offset = self.options['c1_kwidth']
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapC1ToRetina(self, x):
    return self.MapS1ToRetina(self.MapC1ToS1(x))

  def MapC1ToImage(self, x):
    return self.MapRetinaToImage(self.MapC1ToRetina(x))

  def MapS2ToC1(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapS2ToC1
    offset = self.options['s2_kwidth']
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapS2ToImage(self, x):
    return self.MapC1ToImage(self.MapS2ToC1(x))

  def MapC2ToS2(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapC2ToS2
    offset = self.options['c2_kwidth']
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapC2ToImage(self, x):
    return self.MapS2ToImage(self.MapC2ToS2(x))


#### OPTION HANDLING ###########

def _GetOpt(options, key):
  """Lookup the given option, returning a default value if not found."""
  global _DEFAULT_OPTIONS
  if key in options:
    return options[key]
  return _DEFAULT_OPTIONS[key]

def _MapOpts(options, **key_map):
  """Get a set of keyed values from the options dictionary, giving the values
  the desired key in the output dictionary. In the map, the key is the option
  name, while the value is the function parameter name."""
  return dict([ (param, _GetOpt(options, opt_key)) for param, opt_key in
      key_map.items() ])

ALL_OPTIONS = [
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

def MakeDefaultOptions(**options):
  """Make a complete set of default options for a transformation."""
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

    s2_beta = 1.0,
    s2_bias = 1.0,
    s2_kwidth = 7,
    s2_scaling = 2,

    c2_kwidth = 3,
    c2_scaling = 2,

    num_scales = 4,
    scale_factor = 2**(1./2),
    sse_enabled = c_src.GetUseSSE(),
  )

def ExpandOptions(options):
  """Return a full set of options, where each value is taken from the given
  dictionary (if available), or from the set of default options (otherwise)."""
  base_options = MakeDefaultOptions()
  if options == None:
    return base_options
  return dict(base_options.items() + options.items())

_DEFAULT_OPTIONS = MakeDefaultOptions()

def LoadOptions(fname, expand_options = True):
  """Loads option data either from a python file (if fname ends in ".py"), or a
  pickled dictionary.
  fname -- name of file from which to load options
  base_options -- set of default options, which should be overridden by those
                  loaded from input file
  """
  options = util.LoadByFileName(fname)
  if expand_options:
    options = ExpandOptions(options)
  return options

def ApplyGlobalOptions(options):
  """Apply all global option entries."""
  c_src.SetUseSSE(_GetOpt(options, 'sse_enabled'))

######## GENERAL TRANSFORMATION FUNCTIONS ########

def ScaleImage(img, scale, options):
  """Down-sample the given image
  img -- (Image object)
  """
  if scale == 0:
    return img
  num_scales = _GetOpt(options, 'num_scales')
  assert scale < num_scales, "expected scale in [0, %d]: found %d" % (
      num_scales - 1, scale)
  factor = _GetOpt(options, 'scale_factor')
  ratio = factor ** scale
  shape = [int(x / ratio) for x in img.size]
  return img.resize(shape, Image.BICUBIC) #ANTIALIAS)

def BuildRetinaFromImage(img, options):
  """img -- (array) pixel data in the range [0, 1]
  RETURNS (array) retinal data in the range [-1, 1]"""
  if _GetOpt(options, 'retina_enabled'):
    return misc.BuildRetinalLayer(img, **_MapOpts(options,
        kwidth = 'retina_kwidth', bias = 'retina_bias'))
  # print >>sys.stderr, "WARN: retinal layer disabled"
  return img * 2 - 1

def BuildRetinaFromFile(fname, options):
  """RETURNS: image and retina arrays"""
  image = misc.ImageToInputArray(Image.open(fname))
  retina = BuildRetinaFromImage(image, options)
  return image, retina

def BuildS1FromRetina(retina, s1_kernels, options):
  """Apply local RBFs (implemented with normalized dot products) to retinal
  data.
  retina - 2D array of input data
  s1_kernels - 4D array of kernels (2D array of 2D kernels). See
               MakeGaborKernels()
  OPTIONS:
    s1_bias - bias term added to the norm of the input vector
    s1_beta - width of the RBF's Gaussian envelope
    s1_scaling - subsampling factor (e.g., setting s1_scaling=2 will result in
                 an output array that is 1/2 the width and half the height of
                 the input array)
    s1_sparsify -
  RETURNS: 4D array of S1 activity
  """
  # Retina shape is [r_height, r_width]
  # Kernel shape is [num_orientations, num_phases, kheight, kwidth]
  # S1 shape is [num_orientations, num_phases, s1_height, s1_width]
  assert len(retina.shape) == 2
  assert len(s1_kernels.shape) == 4
  retina_ = retina.reshape((1,) + retina.shape)
  # Make kernel 3-D, where the third dimension is unity since retina is only 2-D
  kshape = (-1, 1) + s1_kernels.shape[2:]
  s1_kernels_ = s1_kernels.reshape(kshape)
  s1 = misc.BuildSimpleLayer(retina_, kernels = s1_kernels_,
    **_MapOpts(options, bias = 's1_bias', beta = 's1_beta',
        scaling = 's1_scaling')
      )

  if _GetOpt(options, 's1_sparsify'):
    beta = _GetOpt(options, 's1_beta')
    zero = math.exp(-2 * beta)  # The RBF value corresponding to a dot-product
                                # of zero.
    u = np.abs(s1).reshape((-1,) + s1.shape[-2:])
    tau = 0.01  # Threshold chosen so that different features applied to a flat
                # background will both be preserved. Should be able to reduce
                # this by mapping mean of S1 kernels closer to zero.
    s1[ u < u.max(0) - tau ] = zero  # Apply a soft winner-take-all competition

  # Reshape S1 layer
  s1_ = s1.reshape(s1_kernels.shape[:2] + s1.shape[1:])

  # Map [exp(-4*beta), 1] to [0, 1]
  # offset = math.exp(-4 * _GetOpt(options, 's1_beta'))
  # s1_ -= offset
  # s1_ /= 1 - offset

  return s1_

def BuildC1FromS1(s1, options):
  """Apply local max to S1 data.
  s1 - 4D array of S1 activity
  OPTIONS:
    c1_kwidth - width of the (square) max kernel
    c1_scaling - subsampling factor
    c1_sparsify -
    c1_whiten - whether to "whiten" C1 data before returning it. See Whiten().
  RETURNS: 3D array of C1 activity
  """
  assert len(s1.shape) == 4
  norientations, nphases, iheight, iwidth = s1.shape
  c1_whiten = _GetOpt(options, 'c1_whiten')
  c1, max_coords = misc.BuildComplexLayer_PoolLastBand(s1,
      **_MapOpts(options, kwidth = 'c1_kwidth', kheight = 'c1_kwidth',
          scaling = 'c1_scaling')
      )
  if _GetOpt(options, 'c1_sparsify'):
    u = np.abs(c1)
    c1[ u < u.max(0) ] = 0
  if c1_whiten:
    misc.Whiten(c1)
  return c1, max_coords

def BuildC1FromFile(fname, options):
  """Returns image data, retina data, S1 data for each scale, C1 data for each
  scale."""
  image, retina = BuildRetinaFromFile(fname, options)
  s1_kernels = MakeS1Kernels(options)
  num_scales = _GetOpt(options, 'num_scales')
  def GetScales():
    for scale in range(num_scales):
      s1 = BuildS1FromRetina(retina, s1_kernels, options)
      c1, max_coords = BuildC1FromS1(s1, options)
      yield s1, c1
    raise StopIteration
  s1_scales, c1_scales = zip(*GetScales())
  return image, retina, s1_scales, c1_scales

def BuildS2FromC1(c1, s2_kernels, options):
  """Apply local RBF (as normalized dot product) to C1 data.
  c1 - 3D array of C1 activity
  s2_kernels - set of S2 prototypes that have the same dimensions.
  OPTIONS:
    s2_bias - additive term for input window norm
    s2_beta - RBF tuning width
    s2_scaling - subsampling factor
  RETURNS: 3D array of S2 activity
  """
  # C1 shape is [num_orientations, iheight, iwidth]
  # S2 shape is [num_prototypes, oheight, owidth]
  return misc.BuildSimpleLayer(c1, s2_kernels,
      **_MapOpts(options, bias = 's2_bias', beta = 's2_beta',
          scaling = 's2_scaling'))

def BuildC2FromS2(s2, options):
  """Apply local max filter to S2 data.
  s2 - 3D array of S2 activity
  OPTIONS:
    c2_kwidth - width of (square) max kernel. Set c2_kwidth=None to pool over
                the entire spatial extent of the S2 layer (i.e., resulting in
                one value per S2 kernel).
    c2_scaling - subsampling factor
  RETURNS: 3D array of C2 activity
  """
  # S2 shape is [num_prototypes, iheight, iwidth]
  # C2 shape is [num_prototypes]
  return misc.BuildComplexLayer(s2, **_MapOpts(options, kwidth = 'c2_kwidth',
      kheight = 'c2_kwidth', scaling = 'c2_scaling'))

def ApplyGlobalC2Max(c2_per_scale):
  """Pool spatially over C2 data for each band. This is done for each scale
  independently.
  c2_per_scale - list of 3-D arrays of C2 activity, one per scale
  RETURNS: (list) per-band max activation (stored as vector) for each scale, and
           (list) per-band max coordinates (stored as single-element array of
           2-D bitsets) for each scale.
  """
  max_per_scale = []
  max_coords_per_scale = []
  for c2 in c2_per_scale:
    shape = c2.shape
    assert len(shape) == 3
    # Flatten C2 array spatially (one vector per band)
    c2 = c2.reshape(shape[0], -1)
    max_per_scale.append( c2.max(1) )
    # Compute array locations where activity is equal to per-band maximum
    max_coords = np.array([ band == band.max() for band in c2 ])
    # Unflatten location array
    max_coords = max_coords.reshape(shape)
    max_coords = bitset.FromArray(max_coords,
        2)  # number of dimensions for bitset (max over 2-D spatial location)
    max_coords_per_scale.append(max_coords)
  return max_per_scale, max_coords_per_scale

def BuildItFromGlobalC2Max(max_per_scale):
  """Pool scale-wise over spatial max of C2 data for each band.
  max_per_scale - per-band max activations returned by ApplyGlobalC2Max()
  RETURNS: (vector) per-band max activations, and (1-D array of 1-D bitsets)
           per-band max locations
  """
  max_per_scale = np.array(max_per_scale, activation_dtype)
  max_over_scale = max_per_scale.max(0)
  max_coords_over_scale = np.array([ scale == scale.max() for
      scale in max_per_scale ])
  max_coords_over_scale = bitset.FromArray(max_coords_over_scale,
      1)  # number of dimensions for bitset data (max over 1-D scales)
  return max_over_scale, max_coords_over_scale

def MakeS1Kernels(options):
  """Create Gabor kernels to be used as S1 filters."""
  args = _MapOpts(options,
      num_orientations = 's1_num_orientations', num_phases = 's1_num_phases',
      kwidth = 's1_kwidth', #fscale = 's1_fscale',
      shift_orientations = 's1_shift_orientations')
  args['scale_norm'] = True
  return misc.MakeGaborKernels(**args)

def DrawGaborAsLine(orient, options, **args):
  args = util.MergeDict(args, **_MapOpts(options,
      num_orientations = 's1_num_orientations', kwidth = 's1_kwidth',
      shift_orientations = 's1_shift_orientations'))
  return misc.DrawGaborAsLine(orient, **args)

def ImprintS2Prototypes(img_fname, coords, options):
  """Imprint a set of S2 prototypes from specific regions of C1 activity, which
  is computed from the given image."""
  num_prototypes = len(coords)
  num_orientations = _GetOpt(options, 's1_num_orientations')
  kwidth = _GetOpt(options, 's2_kwidth')
  image, retina, s1_scales, c1_scales = BuildC1FromFile(img_fname, options)
  prototypes = np.empty([num_prototypes, num_orientations, kwidth, kwidth],
      activation_dtype)
  mapper = RegionMapper(options)
  for (scale, y, x), proto in zip(coords, prototypes):
    # Copy C1 activity to kernel array
    y = mapper.MapS2ToC1(slice(y, y+1))
    x = mapper.MapS2ToC1(slice(x, x+1))
    patch = c1_scales[scale][:, y, x]
    proto[:] = patch
    array.ScaleUnitNorm(proto)
  assert np.allclose(np.array(map(np.linalg.norm, prototypes)), 1), \
    "Internal error: S2 prototypes are not normalized"
  assert not np.isnan(prototypes).any(), \
      "Internal error: found NaN in imprinted prototype."
  return prototypes

def ImprintRandomS2Prototypes(img_fname, num_prototypes, options):
  img, retina, s1_scales, c1_scales = BuildC1FromFile(img_fname, options)
  num_scales = _GetOpt(options, 'num_scales')
  num_orientations = _GetOpt(options, 's1_num_orientations')
  kwidth = _GetOpt(options, 's2_kwidth')
  prototypes = np.empty([num_prototypes, num_orientations, kwidth, kwidth],
      activation_dtype)
  for proto in prototypes:
    scale = random.randint(0, num_scales - 1)
    c1 = c1_scales[scale]
    height, width = c1.shape[-2:]
    y = random.randint(0, height - kwidth)
    x = random.randint(0, width - kwidth)
    # Copy C1 activity to kernel array
    proto[:] = c1[:, y : y + kwidth, x : x + kwidth]
    array.ScaleUnitNorm(proto)
  assert np.allclose(np.array(map(np.linalg.norm, prototypes)), 1), \
    "Internal error: S2 prototypes are not normalized"
  assert not np.isnan(prototypes).any(), \
      "Internal error: found NaN in imprinted prototype."
  return prototypes


def _SingleScaleThroughC1(img, layers, options, s1_kernels, use_timer):
  """Transform a single image up to (and including) the C1 layer for a single
  image scale.
  RETURNS: (dict) activity maps for image, retina, S1, and C1 layers, and
           location of maximum activity for C1 layer.
  """
  results = dict()
  if 'r' not in layers: return results
  with util.Timer("img", use_timer):
    img = misc.ImageToInputArray(img)
  results['image'] = img
  with util.Timer("retina", use_timer):
    retina = BuildRetinaFromImage(img, options)
  results['r-activity'] = retina
  if 's1' not in layers: return results
  with util.Timer("s1", use_timer):
    s1 = BuildS1FromRetina(retina, s1_kernels, options)
  results['s1-activity'] = s1
  if 'c1' not in layers: return results
  with util.Timer("c1", use_timer):
    c1, c1_coords = BuildC1FromS1(s1, options)
  results['c1-activity'] = c1
  results['c1-coords'] = c1_coords
  return results

def _SingleScaleThroughIT(img, layers, options, s1_kernels, all_s2_prototypes,
    use_timer):
  """Transform a single image up to (and including) the IT layer for a single
  scale.
  RETURNS: (dict) activity maps for image, retina, S1, C1, S2, and C2 layers,
           and location of maximum activity for C1 and C2 layers. S2 and C2
           layer activity are lists organized by S2 prototype size.
  """
  results = _SingleScaleThroughC1(img, layers, options, s1_kernels, use_timer)
  if 's2' not in layers: return results
  c1 = results['c1-activity']
  results['s2-activity'] = []
  if 'c2' in layers:
    results['c2-activity'] = []
    results['c2-coords'] = []
  for psize in range(len(all_s2_prototypes)):
    s2_prototypes = all_s2_prototypes[psize]
    with util.Timer("Proto Size %s" % psize, use_timer):
      with util.Timer("s2", use_timer):
        s2 = BuildS2FromC1(c1, s2_prototypes, options)
      results['s2-activity'].append(s2)
      if 'c2' not in layers:
        continue
      c2, c2_coords = BuildC2FromS2(s2, options)
      results['c2-activity'].append(c2)
      results['c2-coords'].append(c2_coords)
  return results

def _GlobalC2Max(c2_per_scale):
  """Pool spatially over C2 data for each band. This is done for each scale
  independently.
  c2_per_scale - list of 3-D arrays of C2 activity, one per scale
  RETURNS: (list) per-band max activation (stored as vector) for each scale, and
           (list) per-band max coordinates (stored as single-element array of
           2-D bitsets) for each scale.
  """
  max_per_scale = []
  max_coords_per_scale = []
  for c2_per_psize in c2_per_scale:
    assert (np.array([ len(c2.shape) for c2 in c2_per_psize ]) == 3).all()
    shapes = np.array([ c2.shape for c2 in c2_per_psize ])
    # Flatten C2 array spatially (one vector per band)
    flat_c2s = [ c2.reshape(shape[0], -1) for c2, shape
        in zip(c2_per_psize, shapes) ]
    max_per_psize = np.concatenate([ c2.max(1) for c2 in flat_c2s ])
    # Compute array locations where activity is equal to per-band maximum
    max_coords_per_psize = [
      bitset.FromArray(np.array([ band == band.max() for band in c2 ])
          .reshape(shape), 2)
      for c2, shape in zip(flat_c2s, shapes) ]
    max_per_scale.append(max_per_psize)
    max_coords_per_scale.append(max_coords_per_psize)
  max_per_scale = np.array(max_per_scale, activation_dtype)
  return max_per_scale, max_coords_per_scale

def _ItFromGlobalC2Max(max_per_scale):
  """Pool scale-wise over spatial max of C2 data for each band.
  max_per_scale - per-band max activations returned by _GlobalC2Max()
  RETURNS: (vector) per-band max activations, and (1-D array of 1-D bitsets)
           per-band max locations
  """
  max_over_scale = max_per_scale.max(0)
  max_coords_over_scale = np.array([ band == band.max() for band
      in max_per_scale.T ]).T
  max_coords_over_scale = bitset.FromArray(max_coords_over_scale,
      1)  # number of dimensions for bitset data (max over 1-D scales)
  return max_over_scale, max_coords_over_scale

def _MultiScaleThroughIT(img, layers, options, all_s2_prototypes,
    store_callback, use_timer = False):
  num_scales = _GetOpt(options, 'num_scales')
  imgs = [ ScaleImage(img, scale, options) for scale in range(num_scales) ]
  store_callback("", dict(options = options))
  s1_kernels = None
  if 's1' in layers:
    s1_kernels = MakeS1Kernels(options)
    store_callback("", {'s1-kernels' : s1_kernels})

  def ProcessScales():
    for scale in range(num_scales):
      img = imgs[scale]
      with util.Timer("Scale %d" % scale, use_timer):
        try:
          results = _SingleScaleThroughIT(img, layers, options, s1_kernels,
              all_s2_prototypes, use_timer)
        except misc.InsufficientSizeException:
          # If even the full image is too small, throw an exception
          assert scale > 0, "Image too small"
#          print >>sys.stderr, ("WARN: Unable to process scale [%d]" % scale) + \
#              " -- image too small"
          break  # stop iterating over scales
      store_callback("scale%d-" % scale, results)
      yield results
    raise StopIteration

  all_results = list(ProcessScales())
  if 'it' not in layers: return None
  c2_per_scale = [ r['c2-activity'] for r in all_results ]
  global_max, global_max_coords = _GlobalC2Max(c2_per_scale)
  for s, m, c in zip(range(len(global_max)), global_max, global_max_coords):
    store_callback("scale%d-" % s, {'global-c2-activity' : m,
        'global-c2-coords' : c})
  it, it_coords = _ItFromGlobalC2Max(global_max)
  store_callback("", {'it-activity' : it, 'it-coords' : it_coords,
      'feature-vector' : it})
  return it

def TransformImageFromFile(img_fname, layer, s2_prototypes_fname = None,
    storage_callback = None, options = None, use_timer = False):
  """Transform an image through the given layer. This is intended to be called
  as the main function from a command-line script."""
  options = ExpandOptions(options)
  assert layer in LAYERS
  layers = LAYERS[ : LAYERS.index(layer)+1 ]
  protos = None
  if s2_prototypes_fname != None:
    protos = util.Load(s2_prototypes_fname)
    if not isinstance(protos, list):
      protos = [ protos ]
    ntheta = options['s1_num_orientations']
    for p in protos:
      assert len(p.shape) == 4 and p.shape[1] == ntheta, \
          "S2 prototypes have wrong shape"
      assert np.allclose(np.array(map(np.linalg.norm, p)), 1), \
          "S2 prototypes are not normalized"
      assert not np.isnan(p).any(), \
          "Internal error: found NaN in S2 prototypes."
  elif LAYER_S2 in layers:
    raise util.UsageException("S2 prototypes required to transform through S2 "
        "layer")
  ApplyGlobalOptions(options)
  if storage_callback == None:
    storage_callback = lambda x, y: None

  def _LoadImage(img_fname):
    max_img_size = 512
    img = Image.open(img_fname).convert("L")
    w, h = img.size
    m = min(w, h)
    if m > max_img_size:
      r = max_img_size / float(m)
      w = int(w * r)
      h = int(h * r)
      size = w, h
      print "WARN: down-sampling original image to size (w, h) = %s" % (size,)
      img = img.resize(size)
    return img

  img = _LoadImage(img_fname)
  with util.Timer("Total Time", use_timer):
    _MultiScaleThroughIT(img, layers, options, protos, storage_callback,
        use_timer)

class Results:
  """An accessor for a set of Glimpse results, stored in a single directory."""
  def __init__(self, rdir, image = None, s2_kernels = None):
    """Create a new accessor for a given set of results.
    rdir -- directory containing result files
    image -- path to input image for transform
    s2_kernels -- path to file of S2 prototypes
    """
    self._rdir = rdir
    self._img_fname = image
    self._s2_kernels_fname = s2_kernels
    self._d = {}

    def MakeLoader(layer, dtype):
      # First argument to lambda function is object reference, since this is a
      # a new method attribute.
      name = "%s-%s" % (layer, dtype)
      return lambda self, scale = None: self._load_multiscale(name, scale)

    for y in ('r', 's1', 'c1', 's2', 'c2'):
      name = '%s_activity' % y
      p = property(MakeLoader(y, 'activity'))
      setattr(self.__class__, name, p)
    for y in ('c1', 'c2'):
      name = '%s_coords' % y
      p = property(MakeLoader(y, 'coords'))
      setattr(self.__class__, name, p)

  def _load(self, fname):
    if fname not in self._d:
      self._d[fname] = util.Load(os.path.join(self._rdir, fname))
    return self._d[fname]
  def _load_multiscale(self, dtype, scale = None):
    if scale == None:
      return [ self._load_multiscale(dtype, s) for s in
          range(self.options['num_scales']) ]
    return self._load('scale%d-%s' % (scale, dtype))
  @property
  def options(self):
    return self._load('options')
  @property
  def image(self):
    assert self._img_fname != None
    key = 'image'
    if key not in self._d:
      self._d[key] = misc.ImageToInputArray(Image.open(self._img_fname))
    return self._d[key]
  @property
  def s1_kernels(self):
    return self._load('s1-kernels')
  @property
  def s2_kernels(self):
    assert self._s2_kernels_fname
    key = 's2-kernels'
    if key not in self._d:
      self._d[key] = util.Load(self._s2_kernels_fname)
    return self._d[key]
  @property
  def it_activity(self):
    return self._load('it-activity')
  @property
  def it_coords(self):
    return self._load('it-coords')
  @property
  def feature_vector(self):
    return self._load('feature-vector')
