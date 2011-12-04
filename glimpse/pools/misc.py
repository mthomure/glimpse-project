# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import multiprocessing

class IPool(object):
  """The interface of a worker pool interface."""

  def map(self, func, iterable, chunksize = None):
    raise NotImplemented

  def imap(self, func, iterable, chunksize = 1):
    raise NotImplemented

  def imap_unordered(self, func, iterable, chunksize = 1):
    raise NotImplemented

class SinglecorePool(object):

  def map(self, func, iterable, chunksize = None):
    return map(func, iterable)

  def imap(self, func, iterable, chunksize = 1):
    return map(func, iterable)

  def imap_unordered(self, func, iterable, chunksize = 1):
    return map(func, iterable)

class MulticorePool(object):
  """Thin wrapper around multiprocessing.Pool that supports serialization. If
  serialization is not required, it is easier to use the base class directly."""

  def __init__(self, *args):
    """Create new object. See multiprocessing.Pool() for explanation of
    arguments."""
    # Save the initialization arguments, so we can serialize and then
    # reconstruct this object later.
    self._init_args = args
    self.pool = multiprocessing.Pool(*args)

  def __reduce__(self):
    return (MulticorePool, self._init_args)

  def map(self, func, iterable, chunksize = None):
    return self.pool.map(func, iterable, chunksize)

  def imap(self, func, iterable, chunksize = 1):
    return self.pool.imap(func, iterable, chunksize)

  def imap_unordered(self, func, iterable, chunksize = 1):
    return self.pool.imap_unordered(func, iterable, chunksize)

def MakePool():
  """Return an instance for the best available worker pool.
  RETURN a serializable worker pool
  """
  return MulticorePool()
