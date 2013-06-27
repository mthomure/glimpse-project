# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from glimpse.util.progress import ProgressBar

def _pmap_sync(stream, pb):
  results = list()
  for i,r in enumerate(stream):
    pb.update(i+1)
    results.append(r)
  pb.finish()
  return results

def _pmap_async(stream, pb):
  for i,r in enumerate(stream):
    pb.update(i+1)
    yield r
  pb.finish()

def pmap(view, fn, *xss, **kw):
  """

  Note that pmap will not show incremental progress unless `block` is
  False.

  """
  progress = kw.pop('progress', None)
  assert len(xss) > 0, "No inputs specified"
  if progress is None or progress == False:
    return view.map(fn, *xss, **kw)
  elif progress == True:
    progress = ProgressBar
  block = kw.pop('block', True)
  maxval = None
  if hasattr(xss[0], '__len__'):
    maxval = len(xss[0])
  pb = progress(maxval=maxval).start()
  pb.update(0)
  # Map must be non-blocking to allow incremental progress updates.
  stream = view.map(fn, *xss, block=False, **kw)
  # Result is a list
  if block:
    return _pmap_sync(stream, pb)
  # Result is an iterator
  return _pmap_async(stream, pb)

import cPickle as pickle
import sys

def _load_stream(fh):
  src = pickle.Unpickler(fh)
  while True:
    try:
      obj = src.load()
      yield obj
    except EOFError:
      break

def _open_wrapper(fname, fn):
  """Safe file-open wrapper around _load_stream."""
  with open(fname, 'rb') as fh:
    for x in fn(fh):
      yield x

def load(fname=None, stream=False):
  """Reconstruct serialized data.

  :param fname: File from which to read (default is stdin).
  :type fname: file or str
  :param bool stream: Whether to read multiple objects from the file.
  :return: First object read from file (if `stream` is False), or generator
     producing all objects available in file (otherwise).

  """
  if not fname:
    fname = sys.stdin
  if hasattr(fname, 'read'):
    if stream:
      return _load_stream(fname)
    return pickle.load(fname)
  if stream:
    return _open_wrapper(fname, _load_stream)
  with open(fname, 'rb') as fh:
    return pickle.load(fh)

def _store(obj, fh, stream):
  dst = pickle.Pickler(fh, protocol=-1)
  if stream:
    for o in obj:
      dst.dump(o)
  else:
    dst.dump(obj)

def store(obj, fname=None, stream=False):
  """Serialize an object.

  :param obj: Object to serialize.
  :param fname: File to which data is written (default is stdout).
  :type fname: file or str
  :param bool stream: Whether to write elements of `obj` separately. This
     requires `obj` to be iterable.

  """
  if not fname:
    fname = sys.stdout
  if hasattr(fname, 'write'):
    _store(obj, fname, stream)
  with open(fname, 'wb') as fh:
    _store(obj, fh, stream)
