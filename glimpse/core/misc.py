
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Miscelaneous functions for transforming one layer of a feed-forward hierarchy
# to another.
#

import c_src as c_src
from glimpse.util import ACTIVATION_DTYPE
from c_src import MaxOutputDimensions, DotProduct, NormDotProduct, NormRbf, \
                  Rbf, ContrastEnhance
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
  return np.empty(shape, ACTIVATION_DTYPE)

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
        dtype = ACTIVATION_DTYPE)
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

def BuildRetinalLayer(img, kwidth = 15, kheight = None, bias = 1.0):
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

