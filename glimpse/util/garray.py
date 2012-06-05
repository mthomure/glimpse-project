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
    array = array / (2 * max_val)  # makes copy of input
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

def FlattenLists(data):
  """Flatten nested lists to a single list.

  :param data: Input data.
  :type data: N-dimensional list
  :returns: Flattened copy of input data.
  :rtype: list

  """
  # From: http://stackoverflow.com/questions/2158395
  result = []
  for el in data:
    if hasattr(el, "__iter__") and not isinstance(el, basestring):
      result.extend(FlattenLists(el))
    else:
      result.append(el)
  return result

def FlattenArrays(data):
  """Flatten a nested list of arrays to a one-dimensional vector.

  Note: All arrays should have the same element type.

  :param data: Input data.
  :type data: N-dimensional list of ndarray
  :returns: Flattened copy of input data.
  :rtype: 1D ndarray

  """
  data = FlattenLists(data)
  assert len(data) > 0
  dtype = data[0].dtype
  assert all(x.dtype == dtype for x in data)
  out_size = sum(x.size for x in data)
  out = np.empty((out_size,), dtype)
  offset = 0
  for subdata in data:
    out[offset : offset + subdata.size] = subdata.flat
    offset += subdata.size
  return out

#: .. deprecated:: 0.1.2
#:    Use :func:`FlattenArrays` instead.
ArrayListToVector = FlattenArrays

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
