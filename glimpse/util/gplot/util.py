# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

"""Functions for summarizing matrix datasets with repeated observations."""
import numpy as np

def xs(data):
  """Returns set of unique x-values."""
  if data.ndim == 1:
    # assume rec array
    names = data.dtype.names
    assert names and len(names) == 2, "Assuming record array with 2 fields"
    return np.unique(data[names[0]])  # assume non-numeric data is pre-sorted
  results = np.unique(data[:,0])
  results.sort()
  return results

def ys(data):
  """Returns each set of records.

  :rtype: list of array
  :return: Array of y-values for each unique x-value. Use :func:`xs` to get
     corresponding list of x-values.

  """
  if data.ndim == 1:
    # assume rec array
    names = data.dtype.names
    assert names and len(names) == 2, "Assuming record array with 2 fields"
    return [ data[names[1]][data[names[0]] == x] for x in xs(data) ]
  # Return value is list of list (not 2D array) since sub-lists lengths vary.
  return [ data[data[:,0] == x][:,1] for x in xs(data) ]

def ys_mean(data):
  """Returns the mean of each set of records.

  Record sets are ordered according to :func:`xs`.

  """
  return np.array(map(np.mean, ys(data)))

def ys_std(data):
  """Returns the standard deviation for each set of records.

  Record sets are ordered according to :func:`xs`.

  """
  return np.array(map(np.std, ys(data)))

def ys_stderr(data):
  """Returns the standard error for each set of records.

  Record sets are ordered according to :func:`xs`.

  The standard error is defined as :math:`\frac{s}{\sqrt{n}}` where
  :math:`s` is the standard deviation of the sample, and :math:`n` is the
  number of elements in that sample.

  """

  return np.array([ np.std(y) / np.sqrt(len(y)) for y in ys(data) ])
