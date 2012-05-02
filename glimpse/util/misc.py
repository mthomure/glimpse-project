"""Miscellaneous functions that do not belong in one of the other modules."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import garray
import gimage
import itertools
import math
import numpy
import os
import sys

def TypeName(x):
  """Get the fully-qualified name of an object's type.

  If the argument is a type object, its fully-qualified name is returned.

  :rtype: str

  """
  # If we're already given a type object, then return it's name.
  if isinstance(x, type):
    type_ = x
  else:
    type_ = type(x)
  module = type_.__module__
  # Ignore the builtin namespace.
  if type_.__module__ == '__builtin__':
    return type_.__name__
  return "%s.%s" % (module, type_.__name__)

def MergeDict(new_dict, **base_dict):
  """Merge two dictionaries, with entries in the first dictionary taking
  precedence.

  """
  return dict(base_dict.items() + new_dict.items())

def IsIterable(obj):
  """Determine if an object can be iterated."""
  return hasattr(obj, '__iter__')

def IsString(obj):
  """Determine if an object is a string."""
  return isinstance(obj, basestring)

def IsScalar(v):
  """Determine if an object has scalar type.

  This is presently defined as either an integer, a float, or a string.

  """
  return isinstance(v, int) or \
         isinstance(v, float) or \
         isinstance(v, str)

def TakePairs(lst):
  """Convert a list of values into a list of 2-tuples.

  Example usage::

     >>> TakePairs([1, 2, 3, 4])
     [(1, 2), (3, 4)]

  """
  return [ (lst[i],lst[i+1]) for i in range(0, len(lst), 2) ]

def TakeTriples(lst):
  """Convert a list of values into a list of 3-tuples.

  Example usage::

     >>> TakeTriples([1, 2, 3, 4, 5])
     [(1, 2, 3), (4, 5)]

  """
  return [ (lst[i],lst[i+1],lst[i+2]) for i in range(0, len(lst), 3) ]

def GroupIterator(elements, group_size):
  """Create an iterator that returns sub-groups of an underlying iterator.

  For example, using a group size of three with an input of the first seven
  natural numbers will result in the elements (0, 1, 2), (3, 4, 5), (6,). Note
  that tail elements are not ignored.

  :param elements: An iterable sequence of input values.
  :param int group_size: Number of elements in each returned sub-group.
  :returns: Iterator over sub-groups.

  Example usage::

     >>> GroupIterator([1, 2, 3, 4, 5, 6, 7, 8], 3)
     [(1, 2, 3), (4, 5, 6), (7, 8)]

  """
  element_iter = iter(elements)
  while True:
    batch = tuple(itertools.islice(element_iter, group_size))
    if len(batch) == 0:
      raise StopIteration
    yield batch

def UngroupIterator(groups):
  """Create an iterator that returns each element from each group, one element
  at a time.

  This is the inverse of :func:`GroupIterator`.

  :param groups: A list of iterators, where each iterator may have a different
     lengths.
  :returns: A single iterator that returns all the elements from the first input
     iterator, then all the elements of the second iterator, and so on.

  Example usage::

     >>> UngroupIterator([(1, 2, 3), (4,), (5, 6, 7, 8)])
     [1, 2, 3, 4, 5, 6, 7, 8]

  """
  return itertools.chain(*groups)

def UngroupLists(groups):
  """Concatenate several sequences to form a single list.

  :rtype: list

  .. seealso::
     :func:`UngroupIterator`.

  """
  return list(UngroupIterator(groups))

def SplitList(data, sizes = []):
  """Break a list into unequal-sized sublists.

  :param list data: Input data.
  :param sizes: Size of each chunk. If sum of sizes is less than entire size of
     input array, the remaining elements are returned as an extra sublist in the
     result.
  :type sizes: list of int
  :returns: Sublists of requested size.
  :rtype: list of list

  .. seealso::
     :func:`UngroupIterator`

  """
  assert(all([ s >= 0 for s in sizes ]))
  if len(sizes) == 0:
    return data
  if sum(sizes) < len(data):
    sizes = list(sizes)
    sizes.append(len(data) - sum(sizes))
  out = list()
  last = 0
  for s in sizes:
    out.append(data[last : last+s])
    last += s
  return out

def ToArray(obj):
  """Convert an object to a numpy array.

  :param obj: Input data.
  :type obj: ndarray, PIL.Image, or any object with a `ToArray()` method.
  :rtype: ndarray

  """
  if isinstance(obj, numpy.ndarray):
    return obj
  if hasattr(obj, "load"):
    return gimage.ImageToArray(obj)
  try:
    return obj.ToArray()
  except AttributeError:
    pass
  raise TypeError("Don't know how to create array from %s" % type(obj))

def ToImage(obj):
  """Convert an object to an image.

  :param obj: Input data.
  :type obj: ndarray, PIL.Image, or any object with a `ToArray()` method.
  :rtype: PIL.Image

  """
  if hasattr(obj, "load"):
    return obj
  obj = ToArray(obj)
  return garray.ArrayToGreyscaleImage(obj)

def Show(obj, fname = None):
  """Display an object on the screen.

  :param obj: Input data.
  :type obj: ndarray or PIL.Image
  :param str fname: Attempt to use this string as the name of the temporary
     image file.

  """
  ToImage(obj).show(fname)

class UsageException(Exception):
  """An exception indicating that a program was called inappropriately.

  For example, this could be thrown if a program was passed an invalid command
  line argument.

  """
  def __init__(self, msg = None):
    self.msg = msg

def Usage(msg, exc = None):
  """Print the usage message for a program and exit.

  :param str msg: Usage message.
  :param exc: Exception information to include in printed message.

  """
  if exc and type(exc) == UsageException and exc.msg:
    print >>sys.stderr, exc.msg
  msg = "usage: %s %s" % (os.path.basename(sys.argv[0]), msg)
  # print >>sys.stderr, msg
  sys.exit(msg)

def GetOptions(short_opts, long_opts = (), args = None):
  """Parse command line arguments, raising a UsageException if an error is
  found.

  :param str short_opts: Set of single-character argument keys (see
     documentation for the :mod:`getopt` module).
  :param long_opts: Set of multi-character argument keys.
  :type long_opts: list of str
  :param args: Command line arguments to parse. Defaults to :attr:`sys.argv`.
  :type args: list of str
  :returns: Parsed options, and remaining (unparsed) arguments.
  :rtype: 2-tuple, where the first element is a list of pairs of str, and the
     second element is a list of str.

  """
  import getopt
  if args == None:
    args = sys.argv[1:]
  try:
    opts, args = getopt.getopt(args, short_opts, long_opts)
  except getopt.GetoptError,e:
    raise UsageException(str(e))
  return opts, args

def Blackman1d(n, alpha = 0.16):
  """The 1-dimensional Blackman window.

  .. deprecated:: 0.1
     Use :func:`numpy.blackman` instead.

  :param int n: Number of elements.
  :param float alpha: The parameter of the Blackman window.
  :rtype: 1D ndarray

  """
  a0 = (1 - alpha) / 2.0
  a1 = 0.5
  a2 = alpha / 2.0
  x = numpy.arange(n)
  return a0 - a1 * numpy.cos(2 * math.pi * x / (n - 1)) + \
      a2 * numpy.cos(4 * math.pi * x / (n - 1))

def Blackman2d(ny, nx, power = 1):
  """The 2-dimensional Blackman window.

  :param int ny: Number of elements along the Y-axis.
  :param int nx: Number of elements along the X-axis.
  :param float power: Elongates the X-direction (if greater than 1), or shortens
     it (if less than 1).
  :rtype: 2D ndarray

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

  :param float sigma: Variance of the Gaussian function.
  :param float theta: Orientation of the parallel stripes of the Gabor function.
  :param float phi: Phase offset (match a black edge on on white background, or
     vice versa).
  :param float gamma: Spatial aspect ratio. set to 1.0 to get circular variance
     on the Gaussian function, and set to less than one to get elongated
     variance.
  :param float lambda_: Wavelength of sinusoidal function.
  :param kernel: Array in which to store kernel values (must not be None).
  :type kernel: 2D ndarray of float
  :returns: The *kernel* parameter.
  :rtype: 2D ndarray of float

  """
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
