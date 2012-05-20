"""Miscellaneous functions for dealing with numpy arrays."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import Image
import math
import numpy as np

#: Element type for an array of Glimpse activation values.
ACTIVATION_DTYPE = np.float32

def ArgMax(array):
  """Short-hand to find array indices containing the maximum value."""
  return np.transpose(np.nonzero(array == array.max()))

def ArgMin(array):
  """Short-hand to find array indices containing the minimum value."""
  return np.transpose(np.nonzero(array == array.min()))

def ScaleUnitNorm(x):
  """Scale elements of vector (in place), such that result has unit norm."""
  norm = np.linalg.norm(x)
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
  if array.dtype != np.float32:
    array = array.astype(np.float32)
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
  array = np.asarray(array, dtype = np.uint8)
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
  out = np.empty((out_size,), arrays[0].dtype)
  offset = 0
  for a in arrays:
    out[offset : offset + a.size] = a.flat
    offset += a.size
  return out

def PadArray(data, out_shape, cval):
  """Pad the border of an array with a constant value."""
  out_shape = np.array(out_shape)
  in_shape = np.array(data.shape)
  result = np.empty(out_shape)
  result[:] = cval
  begin = ((out_shape - in_shape) / 2.0).astype(int)
  result[ [ slice(b, e) for b, e in zip(begin, begin + in_shape) ] ] = data
  return result

def CropArray(data, out_shape):
  """Remove the border of an array.

  :param data: Input array.
  :type data: ND ndarray
  :param out_shape: Shape of central region to return. If length is less than
     `data.shape`, this is assumed to specify the range in the last axes.
  :type out_shape: list of int
  :rtype: ND ndarray
  :returns: View of the central region of the input array.

  """
  if out_shape == None or len(out_shape) < 1:
    return data
  in_shape = data.shape
  if len(out_shape) < len(in_shape):
    out_shape = data.shape[:-len(out_shape)] + tuple(out_shape)
  elif len(out_shape) > len(in_shape):
    raise ValueError("Shape parameter has wrong format.")
  in_shape = np.array(in_shape)
  out_shape = np.array(out_shape)
  if not np.all(in_shape >= out_shape):
    raise ValueError("Shape parameter is out of range.")
  begin = ((in_shape - out_shape) / 2.0).astype(int)
  return data[ [ slice(b, e) for b, e in zip(begin, begin + out_shape) ] ]

def CompareArrayLists(xs, ys):
  """Compare lists of arrays.

  :param xs: First list of arrays.
  :type xs: iterable of ndarray
  :param ys: Second list of arrays.
  :type ys: iterable of ndarray
  :rtype: bool
  :returns: True if the two arguments have equal elements, otherwise False.

  """
  if isinstance(xs, np.ndarray):
    if not isinstance(ys, np.ndarray):
      return False
    # Make sure we can call np.all().
    if xs.size != ys.size:
      return False
    return np.all(xs == ys)
  # Arguments are iterables. Compare elements.
  if len(xs) != len(ys):
    return False
  return all(CompareArrayLists(x, y) for x, y in zip(xs, ys))
