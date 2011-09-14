
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Miscelaneous functions for transforming one layer of a feed-forward hierarchy
# to another.
#

import c_src as c_src
from c_src import MaxOutputDimensions
from glimpse import util
from glimpse.util import bitset
import Image
import ImageDraw
import math
import numpy as np
import operator

#### Wrappers For c_src Functions ####

# The following functions are used to ensure that array data is contiguous, and
# that data is properly aligned when SSE intrinsics are to be used.

__SSE_VECTOR_WIDTH = 4

def MakeArray(shape):
  return np.empty(shape, c_src.activation_dtype)

def _PrepareSSEArray(data, is_input_array):
  """Prepare array to be passed to a C++ function, ensuring data is aligned in
  memory for SSE operations.
  data - N-D input or output array, to be passed as an argument to a C++
         function
  is_input_array - indicates whether data array is to be used as an input
                   argument. if so, data will be copied if/when a new array is
                   created.
  RETURNS: properly padded N-D array (copied from data argument if necessary),
           and a flag indicating whether this array is newly created.
  """
  remainder = data.shape[-1] % __SSE_VECTOR_WIDTH
  result_is_new = False
  if c_src.GetUseSSE() and remainder > 0:
    width = data.shape[-1]
    if is_input_array:
      # Pad so width is 16-byte aligned
      new_width = width + (__SSE_VECTOR_WIDTH - remainder)
    else:
      # Strip so width is 16-byte aligned
      new_width = width - remainder
    new_data = np.zeros(data.shape[:-1] + (new_width,),
        dtype = c_src.activation_dtype)
    if is_input_array:
      # Copy input array elements to newly-allocated memory
      slices = [ slice(0, x) for x in data.shape ]
      new_data[ slices ] = data[:]
    data = new_data
    result_is_new = True
  return data, result_is_new

def _CopyOutputArray(checked_data, data):
  """Copy output data back to original array."""
  slices = [ slice(0, min(x1, x2)) for x1, x2 in zip(checked_data.shape,
      data.shape) ]
  data[ slices ] = checked_data[ slices ]


#### Generic Transform Functions ####

class InsufficientSizeException(BaseException):
  """Exception indicating that the input array was too small (spatially) to
  support the requested operation."""
  pass

def MakeGaborKernel(kwidth, theta, gamma = 0.6, sigma = None, phi = 0,
    lambda_ = None, scale_norm = True):
  """Create a kernel matrix by evaluating the 2-D Gabor function.
  kwidth - width of the square kernel (should be odd)
  theta - orientation of the (normal to the) preferred frequency
  gamma - aspect ratio of Gaussian window (gamma = 1 means a circular window,
          0 < gamma < 1 means the window is elongated)
  sigma - standard deviation of Gaussian window (1/4 wavelength is good choice,
          so that the window covers approximately two full wavelengths of the
          sine function)
  phi - phase offset (phi = 0 detects white edge on black background, phi = PI
        detects black edge on white background)
  lambda_ - wavelength of sine function (2/5 * kwidth is good choice, so that
            the kernel can fit 2 1/2 wavelengths total)
  scale_norm - if true, then rescale kernel vector to have unit norm
  """
  from numpy import sin, cos, exp, mgrid
  from math import pi
  if lambda_ == None:
    # Allow four cycles of sine wave
    lambda_ = kwidth / 4.0
  if sigma == None:
    # Window should (approximately) go to zero after two wavelengths from center
    sigma = lambda_ / 2.0
  w = kwidth / 2
  Y, X = mgrid[-w:w+1, -w:w+1]
  Yp = -X * sin(theta) + Y * cos(theta)
  Xp = X * cos(theta) + Y * sin(theta)
  k0 = sin(2 * pi * Xp / lambda_ + phi)  # sine wave
  k1 = exp(-(Xp**2 + gamma**2 * Yp**2) / (2.0 * sigma**2))  # Gaussian window
  kernel = k0 * k1  # windowed sine wave
  if scale_norm:
    util.ScaleUnitNorm(kernel)
  return kernel

def MakeGaborKernels(kwidth, num_orientations, num_phases, shift_orientations,
    **args):
  """Create a set of 2-D square kernel arrays whose components are chosen
  according to the Gabor function.
  kwidth - width of kernel (should be odd)
  num_orientations - number of edge orientations (orientations will be spread
                     between 0 and pi)
  num_phases - number of Gabor phases (a value of 2 matches a white edge on a
               black background, and vice versa)
  shift_orientations - whether to rotate Gabor through a small angle (a value of
                       True helps compensate for aliasing)
  scale_norm - if true, then rescale kernel vector to have unit norm
  RETURNS: 4-D array of kernel values
  """
  from math import pi
  if shift_orientations:
    offset = 0.5
  else:
    offset = 0
  thetas = pi / num_orientations * (np.arange(num_orientations) + offset)
  phis = 2 * pi * np.arange(num_phases) / num_phases
  ks = np.array([[ MakeGaborKernel(kwidth, theta, phi = phi, **args) for phi in phis ]
      for theta in thetas ], c_src.activation_dtype)
  return ks

def DrawGaborAsLine(orient, num_orientations = 8, kwidth = 11,
    shift_orientations = False, line_width = 1):
  """Draw the line corresponding to a given Gabor. The generated line is phase
  independent.
  norientations - number of edge orientations
  kwidth - width of kernel in pixels
  shift_orientations - whether to rotate Gabor through a small angle
  line_width - width of drawn line in pixels
  """
  assert orient < num_orientations, "Expected orientation in [0, %s]: got %s" %\
      (num_orientations, orient)
  if shift_orientations:
    theta_shift = 0.5
  else:
    theta_shift = 0
  theta = math.pi / num_orientations * (theta_shift + orient)
  hw = kwidth / 2
  theta_p = theta % math.pi
  if theta_p < math.pi / 4 or theta_p > 3 * math.pi / 4:
    # compute y from x
    x1 = -hw
    x2 = hw
    y1 = math.tan(theta) * x1
    y2 = math.tan(theta) * x2
  else:
    # compute x from y
    y1 = -hw
    y2 = hw
    x1 = y1 / math.tan(theta)
    x2 = y2 / math.tan(theta)
  im = Image.new("L", (kwidth, kwidth), 0)
  draw = ImageDraw.Draw(im)
  draw.line((x1 + hw, y1 + hw, x2 + hw, y2 + hw), fill = 255,
      width = line_width)
  data = util.ImageToArray(im) / 255
  data = np.rot90(data)
  return data

def MakeRandomKernels(nkernels, kshape, normalize = True, mean = 0,
    std = 0.15):
  """Create a set of N-dimensional kernel arrays whose components are sampled
  independently from the normal distribution.
  nkernels - number of kernels to create
  kshape - dimensions of each kernel
  normalize - whether the resulting kernels should be scaled to have unit norm
  mean - center of the component-wise normal distribution
  std - standard deviation of the component-wise normal distribution
  RETURNS: N-D array of kernel values [where N = len(kshape)+1]
  """
  shape = (nkernels,) + kshape
  s2_kernels = np.random.normal(mean, std, shape).astype(np.float32)
  if normalize:
    for k in s2_kernels:
      # Scale vector norm of given array to unity.
      k /= math.sqrt(np.sum(k**2))
  return s2_kernels

def ImageToInputArray(img):
  """Load image into memory in the format required by BuildRetinaFromImage().
  img - PIL Image object containing pixel data
  RETURNS: 2-D array of image data in the range [0, 1]
  """
  img = img.convert('L')
  array = util.ImageToArray(img, transpose = True)
  array = array.astype(np.float32)
  if not array.flags['C_CONTIGUOUS']:
    array = array.copy()
  # Map from [0, 255] to [0, 1]
  array /= 255
  return array

def BuildRetinaFromImage(img, kwidth = 15, kheight = None, bias = 1.0):
  """Enhance local image contrast by transforming each pixel into (local)
  standard-Normal random variables.
  img - 2-D (contiguous) array of pixel data in the range [0, 1]. See
        ImageToInputArray().
  kwidth - width of the local neighborhood over which to compute statistics
  kheight - height of local neighborhood. defaults to value of kwidth.
  bias - term added to standard deviation of local window
  RETURNS: 2-D array of retinal activity
  """
  if kheight == None:
    kheight = kwidth
  oheight, owidth = c_src.MaxOutputDimensions(kheight, kwidth,
      1,  # scaling
      img.shape)[:2]
  if oheight <= 0 or owidth <= 0:
    raise InsufficientSizeException()
  retina = np.zeros((oheight, owidth), np.float32)
  img_, t = _PrepareSSEArray(img, is_input_array = True)
  retina_, is_copy = _PrepareSSEArray(retina, is_input_array = False)
  assert not np.isnan(img_).any()
  c_src.ProcessRetina(img_, kheight, kwidth, bias, retina_)
  assert not np.isnan(retina_).any()
  if is_copy:
    _CopyOutputArray(retina_, retina)
  return retina

def BuildSimpleLayer(idata, kernels, bias = 1.0, beta = 1.0, scaling = 1.0):
  """Apply local RBF (as normalized dot product) to input data.
  idata - 3D (contiguous) array of input activity
  kernels - set of S-unit prototypes that have the same dimensions
  bias - additive term for input window norm
  beta - RBF tuning width
  scaling - subsampling factor
  RETURNS: 3D array of S-unit activity
  """
  # idata shape is [nbands, iheight, iwidth]
  # odata shape is [nkernels, oheight, owidth]
  assert len(idata.shape) == 3
  assert len(kernels.shape) == 4
  kheight, kwidth = kernels.shape[2:]
  assert kheight == kwidth
  # This is a hack -- ProcessSimpleLayer() in c_src.pyx will only use SSE
  # implementation if scaling = 1. Otherwise, we want the non-SSE version of
  # MaxOutputDimensions().
  use_sse = (scaling == 1)
  # BUG: simple-layer processing under current SSE implementation is buggy
  # for some retinal layer sizes. For now, work around by disabling SSE.
  use_sse = False
  oheight, owidth = MaxOutputDimensions(kheight, kwidth, scaling,
      idata.shape[-2:], use_sse)[:2]
  nkernels = kernels.shape[0]
  if oheight <= 0 or owidth <= 0:
    raise InsufficientSizeException()
  odata = np.empty((nkernels, oheight, owidth), np.float32)
  if use_sse:
    idata_, t = _PrepareSSEArray(idata, True)
    odata_, is_copy = _PrepareSSEArray(odata, False)
    f = _PrepareSSEArray
    c_src.ProcessSimpleLayer(idata_, kernels, bias, beta, scaling, odata_, use_sse)
    if is_copy:
      _CopyOutputArray(odata_, odata)
  else:
    c_src.ProcessSimpleLayer(idata, kernels, bias, beta, scaling, odata, use_sse)
  return odata

def BuildComplexLayer(idata, kwidth = None, kheight = None, scaling = 1):
  """Apply local max filter to S2 data.
  idata - 3D (contiguous) array of input activity
  kwidth - width of the max kernel. Set kwidth=None to pool over the horizontal
           spatial extent of the input layer.
  kheight - height of the max kernel. Set kheight=None to pool over the vertical
            spatial extent of the input layer.
  scaling - subsampling factor
  RETURNS: 3D array of C-unit activity
  """
  # idata shape is [nbands, iheight, iwidth]
  # odata shape is [nbands, oheight, owidth]
  assert len(idata.shape) == 3
  nkernels, iheight, iwidth = idata.shape
  if kheight == None:
    # No kernel height specified, pool over vertical extent
    kheight = iheight
  if kwidth == None:
    # No kernel width specified, pool over horizontal extent
    kwidth = iwidth
  use_sse = False
  oheight, owidth = MaxOutputDimensions(kheight, kwidth, scaling,
      (iheight, iwidth), use_sse)[:2]
  if oheight <= 0 or owidth <= 0:
    raise InsufficientSizeException()
  array_shape = (nkernels, oheight, owidth)
  bitset_shape = (kheight, kwidth)
  odata = np.empty(array_shape, np.float32)
  max_coords = bitset.MakeBitsetArray(array_shape, bitset_shape)
  c_src.ProcessComplexLayer(idata, kheight, kwidth, scaling, odata,
      max_coords.memory)
  return odata, max_coords

def BuildComplexLayer_PoolLastBand(idata, kheight = None, kwidth = None,
    scaling = 1):
  """Apply local max filter to S2 data.
  idata - 3D (contiguous) array of input activity
  kwidth - width of the max kernel. Set kwidth=None to pool over the horizontal
           spatial extent of the input layer.
  kheight - height of the max kernel. Set kheight=None to pool over the vertical
            spatial extent of the input layer.
  scaling - subsampling factor
  RETURNS: 3D array of C-unit activity
  """
  # idata shape is [nbands1, nbands2, iheight, iwidth]
  # odata shape is [nbands1, oheight, owidth]
  assert len(idata.shape) == 4
  nbands1, nbands2, iheight, iwidth = idata.shape
  if kheight == None:
    # No kernel height specified, pool over vertical extent
    kheight = iheight
  if kwidth == None:
    # No kernel width specified, pool over horizontal extent
    kwidth = iwidth
  use_sse = False
  oheight, owidth = MaxOutputDimensions(kheight, kwidth, scaling,
      (iheight, iwidth), use_sse)[:2]
  if oheight <= 0 or owidth <= 0:
    raise InsufficientSizeException()
  array_shape = (nbands1, oheight, owidth)
  bitset_shape = (nbands2, kheight, kwidth)
  odata = np.empty(array_shape, np.float32)
  max_coords = bitset.MakeBitsetArray(array_shape, bitset_shape)
  c_src.ProcessC1Layer_PoolSpaceAndPhase(idata, kheight, kwidth, scaling, odata,
      max_coords.memory)
  return odata, max_coords

def Whiten(data):
  """Scale vector norm of every spatial location of array to unity.
  data - 3D array of node activities, which is modified in-place.
  RETURNS: data array
  """
  # Make sure this is a 3-D array with multiple bands
  assert len(data.shape) == 3 and data.shape[0] > 1
  data -= data.mean(0)
  data /= ( np.sqrt((data**2).sum(0)) + 0.00001)
  return data

