"""Miscellaneous functions for dealing with numpy arrays."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import Image
import math
import numpy

#: Element type for an array of Glimpse activation values.
ACTIVATION_DTYPE = numpy.float32

def ArgMax(array):
  """Short-hand to find array indices containing the maximum value."""
  return numpy.transpose(numpy.nonzero(array == array.max()))

def ArgMin(array):
  """Short-hand to find array indices containing the minimum value."""
  return numpy.transpose(numpy.nonzero(array == array.min()))

def ScaleUnitNorm(x):
  """Scale elements of vector (in place), such that result has unit norm."""
  norm = numpy.linalg.norm(x)
#  norm = math.sqrt((x**2).sum())
  if norm == 0:
    x[:] = 1.0 / math.sqrt(x.size)
  else:
    x /= norm
  return x

def ArrayToGreyscaleImage(array, normalize = True):
  """Create a greyscale image from a 2D numpy array.

  This function assumes range of input values contains 0.

  """
  if array.dtype != numpy.float32:
    array = array.astype(numpy.float32)
  if len(array.shape) > 2:
    # Stack bands vertically.
    array = array.reshape((-1, array.shape[-1]))
  if normalize:
    # Normalize array values to lie in [0, 255].
    max_val = max(abs(array.min()), array.max())
    # Map values to lie in [-.5, .5]
    array = array / (2 * max_val)       # makes copy of input
    # Map to [0, 1]
    array += 0.5
    # Map to [0, 255]
    array *= 255
  # Convert to unsigned chars
  array = numpy.asarray(array, dtype = numpy.uint8)
  return Image.fromarray(array, 'L')

def ArrayToRGBImage(array):
  """Create a color image from an array.

  :rtype: PIL.Image

  """
  return Image.fromarray(array, 'RGB')

def ArrayListToVector(arrays):
  """Flatten a list of numpy arrays to a single numpy vector.

  :type arrays: list of ndarray
  :rtype: ndarray

  """
  assert len(arrays) > 0
  out_size = sum(a.size for a in arrays)
  out = numpy.empty((out_size,), arrays[0].dtype)
  offset = 0
  for a in arrays:
    out[offset : offset + a.size] = a.flat
    offset += a.size
  return out

def PadArray(data, out_shape, cval):
  """Pad the border of an array with a constant value."""
  out_shape = numpy.array(out_shape)
  in_shape = numpy.array(data.shape)
  result = numpy.empty(out_shape)
  result[:] = cval
  begin = ((out_shape - in_shape) / 2.0).astype(int)
  result[ [ slice(b, e) for b, e in zip(begin, begin + in_shape) ] ] = data
  return result

def CropArray(data, out_shape):
  """Remove the border of an array.

  :param ndarray data: Input array.
  :param out_shape: Shape of central region to return.
  :type out_shape: list of int
  :rtype: ndarray
  :returns: View of the central region of input array.

  """
  assert numpy.all(numpy.array(data.shape) >= numpy.array(out_shape))
  out_shape = numpy.array(out_shape)
  in_shape = numpy.array(data.shape)
  begin = ((in_shape - out_shape) / 2.0).astype(int)
  return data[ [ slice(b, e) for b, e in zip(begin, begin + out_shape) ] ]
