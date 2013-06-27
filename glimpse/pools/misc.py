# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import importlib
from itertools import imap
import multiprocessing.pool
import os

DEFAULT_POOL_TYPE = "multicore"
DEFAULT_CLUSTER_TYPE = "ipython"

class SinglecorePool(object):
  """A fall-back worker pool that uses a single core of a single machine."""

  def map(self, func, iterable, progress = None):
    """Apply a function to a list using a single local core."""
    if progress is None:
      return map(func, iterable)
    p = progress(len(iterable)).start()
    p.update(0)
    results = imap(func, iterable)
    out = list()
    for i, r in enumerate(results):
      p.update(i+1)
      out.append(r)
    p.finish()
    return out

class MulticorePool(multiprocessing.pool.Pool):
  """A worker pool that uses multiple cores on a single machine."""

  def map(self, func, iterable, progress = None):
    """Apply a function to a list using multiple local cores."""
    if progress is None:
      return super(MulticorePool, self).map(func, iterable)
    iterable = list(iterable)
    if len(iterable) == 0:
      return list()
    p = progress(maxval = len(iterable)).start()
    p.update(0)
    chunksize = 4  # use a small chunksize to support progress updates
    results = self.imap(func, iterable, chunksize = chunksize)
    out = list()
    for i, r in enumerate(results):
      p.update(i+1)
      out.append(r)
    p.finish()
    return out

def MakePool(name=None, **kw):
  """Return a new instance of the given worker pool.

  :param str name: Type of requested pool.
  :param int processes: Number of local cores to use for multicore pool.
     The default is read from `GLIMPSE_MULTICORE_PROCS` environment variable.
  :param str cluster_type: Name of cluster sub-package (e.g., "ipython").
  :param dict cluster_args: Keyword arguments for cluster constructor (default
     is given by evaluating as a python expression the `GLIMPSE_CLUSTER_ARGS`
     environment variable).

  Valid entries for `name` include:

  "s" or "singlecore"
    One core of the local machine are used.
  "m" or "multicore"
    Multiple cores of the local machine are used.
  "c" or "cluster"
    A group of remote machines are used.

  If `name` is unset, the default is read from the `GLIMPSE_POOL` environment
  variable, or else the `DEFAULT_POOL_TYPE` global variable.

  .. seealso::
     :func:`GetClusterPackage` for information about on `cluster_type` argument.

  """
  if name == None:
    name = os.environ.get('GLIMPSE_POOL', DEFAULT_POOL_TYPE)
  else:
    name = name.lower()
  if name in ("s", "singlecore", "singlecore_pool", "singlecorepool"):
    return SinglecorePool()
  if name in ("m", "multicore", "multicore_pool", "multicorepool"):
    if 'processes' not in kw:
      procs = os.environ.get('GLIMPSE_MULTICORE_PROCS')
      if procs is not None:
        kw['processes'] = int(procs)
    return MulticorePool(processes = kw.get('processes'))
  if name in ("c", "cluster"):
    pkg = GetClusterPackage(cluster_type=kw.get('cluster_type'))
    cluster_args = kw.get('cluster_args')
    if cluster_args is None:
      cluster_args = os.environ.get('GLIMPSE_CLUSTER_ARGS')
      if cluster_args is not None:
        cluster_args = eval(cluster_args)
    return pkg.MakePool(**(cluster_args or {}))
  raise ValueError("Invalid pool type: %s" % name)

def GetClusterPackage(cluster_type = None):
  """Choose a cluster package by name.

  :param str cluster_type: Type of cluster to create (e.g., `ipython`). The
     default is read from the `GLIMPSE_CLUSTER` environment variable, or the
     `DEFAULT_CLUSTER_TYPE` if unset.

  If `cluster_type` is "ipython", for example, then this method returns
  :mod:`glimpse.pools.ipythoncluster`. The returned module is guaranteed to
  contain a `MakePool()` function.

  """
  import os
  if cluster_type == None:
    cluster_type = os.environ.get('GLIMPSE_CLUSTER', DEFAULT_CLUSTER_TYPE)
  cluster_mod = importlib.import_module(".%scluster" % cluster_type,
      __package__)
  return cluster_mod
