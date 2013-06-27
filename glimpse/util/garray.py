"""Miscellaneous functions for dealing with numpy arrays."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import Image
import math
import numpy as np
import scipy.misc

def fromimage(im, flatten = 0):
  """Convert Image to numpy array."""
  if isinstance(im, np.ndarray):
    return im  # make function idempotent
  if not Image.isImageType(im):
    raise TypeError("Input is not a PIL image.")
  if flatten:
    im = im.convert('F')
  elif im.mode == '1':
    # Add workaround to bug on converting binary image to numpy array.
    im = im.convert('L')
  elif im.mode == 'LA':
    raise ValueError("Greyscale with alpha channel not supported. Convert to"
        " RGBA first.")
  return np.array(im)

def toimage(arr, *args, **kw):
  if Image.isImageType(arr):
    return arr
  return scipy.misc.toimage(arr, *args, **kw)

FromImage = fromimage
ToImage = toimage

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

def FlattenLists(data):
  """Flatten nested lists to a single list.

  Any arrays are returned unchanged. Thus, this function can be used to turn a
  list of lists (of lists...) of arrays into a single list of arrays.

  :param data: Input data.
  :type data: N-dimensional iterable
  :returns: Flattened copy of input data.
  :rtype: list

  """
  # From: http://stackoverflow.com/questions/2158395
  result = []
  if isinstance(data, np.ndarray):
    return data
  for el in data:
    if hasattr(el, "__iter__") and not (isinstance(el, basestring) or \
        isinstance(el, dict) or isinstance(el, np.ndarray)):
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
  if len(data) == 0:
    return np.empty((0,))
  dtype = data[0].dtype
  if not all(x.dtype == dtype for x in data):
    raise ValueError("All arrays must have the same dtype")
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
  """Pad the border of an array with a constant value.

  :param data: Input data.
  :type data: N-dim ndarray
  :param out_shape: Dimensions of output array.
  :type out_shape: list of int
  :param cval: Value to use for border region of output array.
  :rtype: N-dim ndarray
  :return: Padded input data.

  """
  out_shape = np.array(out_shape)
  in_shape = np.array(data.shape)
  # Ensure the requested size is at least as large as the input array
  for i in range(len(out_shape)):
    out_shape[i] = max(out_shape[i], in_shape[i])
  result = np.empty(out_shape)
  result[:] = cval
  begin = ((out_shape - in_shape) / 2.0).astype(int)
  result[ [ slice(b, e) for b, e in zip(begin, begin + in_shape) ] ] = data
  return result

def CropArray(data, out_shape):
  """Remove the border of an array.

  :param data: Input array.
  :type data: N-dim ndarray
  :param out_shape: Shape of central region to return. If length is less than
     `data.shape`, this is assumed to specify the range in the last axes.
  :type out_shape: list of int
  :rtype: N-dim ndarray
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
