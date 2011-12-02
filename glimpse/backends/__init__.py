# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from cython_backend import CythonBackend
from scipy_backend import ScipyBackend

def ChooseBackend():
  """Return an instance for the best available backend."""
  return CythonBackend()
