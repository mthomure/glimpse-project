
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Miscellaneous functions that do not belong in one of the other modules.
#

from __future__ import absolute_import  # use base "os", not "glimpse.util.os"

import math
import numpy
import os
import sys
from glimpse.util import array
from glimpse.util import image

def MergeDict(new_dict, **base_dict):
  """Merge two dictionaries, with entries in the first dictionary taking
  precedence."""
  return dict(base_dict.items() + new_dict.items())

def IsIterable(obj):
  """Determine if an object can be iterated."""
  return hasattr(obj, '__iter__')

def IsString(obj):
  """Determine if an object has string type."""
  return isinstance(obj, basestring)

def IsScalar(v):
  """Determine if an object has scalar type."""
  return isinstance(v, int) or \
         isinstance(v, float) or \
         isinstance(v, str)

def TakePairs(lst):
  """Convert a list of values into a list of 2-tuples."""
  return [ (lst[i],lst[i+1]) for i in range(0, len(lst), 2) ]

def TakeTriples(lst):
  """Convert a list of values into a list of 3-tuples."""
  return [ (lst[i],lst[i+1],lst[i+2]) for i in range(0, len(lst), 3) ]

def ToArray(obj):
  if isinstance(obj, numpy.ndarray):
    return obj
  if hasattr(obj, "load"):
    return image.ImageToArray(obj)
  try:
    return obj.ToArray()
  except AttributeError:
    pass
  raise TypeError("Don't know how to create array from %s" % type(obj))

def ToImage(obj):
  if hasattr(obj, "load"):
    return obj
  obj = ToArray(obj)
  return array.ArrayToGreyscaleImage(obj)

def Show(obj, fname = None):
  """Display an (array or image) object on the screen.
  fname - Attempt to use this string as the name of the temporary image file."""
  ToImage(obj).show(fname)

def Distribute(total, nbins):
  """Segment a list of objects into seperate bins, such that the distribution of
  objects is as even as possible."""
  bins = numpy.zeros([nbins], numpy.int)
  bins[:] = total / nbins
  remainder = total % nbins
  bins[:remainder] += 1
  return bins

def SplitList(values, num_lists):
  """Split a list of values into equal-width (or as equal width as possible)
  sub-lists."""
  idx = 0
  results = []
  for size in Distribute(len(values), num_lists):
    results.append(values[idx : idx + size])
    idx += size
  return results

class UsageException(Exception):
  """An exception indicating that a program was called inappropriately --- for
  example, by being passed invalid command line arguments."""
  def __init__(self, msg = None):
    self.msg = msg

def Usage(msg, exc = None):
  if exc and type(exc) == UsageException and exc.msg:
    print >>sys.stderr, exc.msg
  msg = "usage: %s %s" % (os.path.basename(sys.argv[0]), msg)
  # print >>sys.stderr, msg
  sys.exit(msg)

def GetOptions(short_opts, long_opts = ()):
  """Parse command line arguments from sys.argv, raising a UsageException if an
  error is found."""
  import getopt
  try:
    opts, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
  except getopt.GetoptError,e:
    raise UsageException(str(e))
  return opts, args


def Blackman1d(n, alpha = 0.16):
  """The 1-dimensional Blackman window.
  n -- number of elements"""
  a0 = (1 - alpha) / 2.0
  a1 = 0.5
  a2 = alpha / 2.0
  x = numpy.arange(n)
  return a0 - a1 * numpy.cos(2 * math.pi * x / (n - 1)) + \
      a2 * numpy.cos(4 * math.pi * x / (n - 1))

def Blackman2d(ny, nx, power = 1):
  """The 2-dimensional Blackman window.
  ny -- number of elements along the Y-axis
  nx -- number of elements along the X-axis
  power -- elongates (if greater than 1) the X-direction, or shortens it (if
           less than 1)
  """
  a = numpy.empty([ny, nx])
  bx = Blackman1d(nx)
  bx = numpy.maximum(bx, numpy.zeros_like(bx)) ** power
  by = Blackman1d(ny)
  a[:] = bx
  a = (a.T * by).T
  return numpy.maximum(a, numpy.zeros_like(a))

def Gabor(sigma, theta, phi, gamma, lambda_, kernel):
  """Fill a 2D matrix using values of the Gabor function.
  sigma -- variance of the Gaussian function
  theta -- orientation of the parallel stripes of the Gabor function
  phi - phase offset (match a black edge on on white background, or vice versa)
  gamma -- spatial aspect ratio. set to 1.0 to get circular variance on the
           Gaussian function, and set to less than one to get elongated
           variance.
  lambda_ -- wavelength of sinusoidal function
  kernel - 2-D array in which to store kernel values (must not be None)
  RETURNS: kernel"""
  height, width = kernel.shape
  size = height * width
  for j in range(height):
    for i in range(width):
      y = j - height / 2.0
      x = i - width / 2.0
      yo = -1.0 * x * math.sin(theta) + y * math.cos(theta)
      xo = x * math.cos(theta) + y * math.sin(theta)
      kernel[j,i] = math.exp(-1.0 *
                      (xo**2 + gamma**2 * yo**2) / (2.0 * sigma**2)) * \
                    math.sin(phi + 2 * math.pi * xo / lambda_)
  return kernel

