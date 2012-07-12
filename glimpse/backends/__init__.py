# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from .cython_backend import CythonBackend
from .scipy_backend import ScipyBackend
from .backend import BackendException, InsufficientSizeException

def MakeBackend(name = None):
  """Return an instance of the given backend.

  :param str name: Name of requested backend.

  """
  if name == None:
    name = "cython"
  else:
    name = name.lower()
  if name in ("cython", "cython_backend", "cythonbackend"):
    return CythonBackend()
  if name in ("scipy", "scipy_backend", "scipybackend"):
    return ScipyBackend()
  raise ValueError("Unknown class name: %s" % (name,))
