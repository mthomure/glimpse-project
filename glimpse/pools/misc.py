# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import multiprocessing

class SinglecorePool(object):
  """A fall-back worker pool that uses a single core of a single machine."""

  def map(self, func, iterable, chunksize = None):
    """Apply a function to a list."""
    return map(func, iterable)

  def imap(self, func, iterable, chunksize = 1):
    """Apply a function to an iterable."""
    return map(func, iterable)

  def imap_unordered(self, func, iterable, chunksize = 1):
    """Apply a function to an iterable, where elements are evaluated in any
    order.

    """
    return map(func, iterable)

class MulticorePool(object):
  """A worker pool that utilizes multiple cores on a single machine.

  This class delegates to the corresponding methods of
  :class:`multiprocessing.Pool`. Thus, all restrictions --- such as
  limitations on mapping an anonymous function --- are preserved.

  Note that there are limitations regarding the functions that can be mapped
  with the methods of this class. Specifically, note that the following
  example will fail::

    >>> pool = glimpse.pools.MulticorePool()
    >>> def f(x): return x * 10
    >>> pool.map(f, [1, 2, 3])
    AttributeError: 'module' object has no attribute 'f'

  Here, the :exc:`AttributeError` is thrown because :func:`f` is not defined
  in the worker subprocess. Similarly, an anonymous function also throws an
  error::

    >>> pool.map((lambda x: x * 10), [1, 2, 3])
    PicklingError: Can't pickle <type 'function'>: attribute lookup
    __builtin__.function failed

  Here, it becomes clear that communication between subprocesses uses
  pickled objects. The :exc:`PicklingError` results from the fact that an
  anonymous function can not be pickled.

  """

  def __init__(self, *args):
    """Create new object.

    See :class:`multiprocessing.Pool` for explanation of arguments.

    """
    # Save the initialization arguments, so we can serialize and then
    # reconstruct this object later.
    self._init_args = args
    self.pool = multiprocessing.Pool(*args)

  def __reduce__(self):
    return (MulticorePool, self._init_args)

  def map(self, func, iterable, chunksize = None):
    """Apply a function to a list."""
    return self.pool.map(func, iterable, chunksize)

  def imap(self, func, iterable, chunksize = 1):
    """Apply a function to an iterable."""
    return self.pool.imap(func, iterable, chunksize)

  def imap_unordered(self, func, iterable, chunksize = 1):
    """Apply a function to an iterable, where elements are evaluated in any
    order.

    """
    return self.pool.imap_unordered(func, iterable, chunksize)

def MakePool():
  """Return a new instance of the default worker pool.

  :returns: A serializable worker pool.

  """
  return MulticorePool()

def GetClusterPackage(cluster_type = None):
  """Choose a cluster package by name.

  If *cluster_type* is `gearman`, for example, then this method returns
  :mod:`glimpse.pools.gearman_cluster`. The returned module is guaranteed to
  contain the ``MakePool()`` and ``RunMain()`` functions.

  :param str cluster_type: Type of cluster to create (e.g., `gearman`, `zmq`,
     `ipython`). The default is read from the ``GLIMPSE_CLUSTER_TYPE``
     environment variable, or is ``ipython`` if this variable is unset.

  """
  import os
  if cluster_type == None:
    cluster_type = os.environ.get('GLIMPSE_CLUSTER_TYPE', 'ipython')
    if cluster_type == None:
      raise Exception('Must specify pool type')
  cluster_mod = __import__("glimpse.pools.%s_cluster" % cluster_type, globals(),
    locals(), ["MakePool", "RunMain"], 0)
  return cluster_mod
