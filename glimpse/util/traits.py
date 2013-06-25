# Traits was recently moved out of the "enthought" package. Support either new
# or old implementation.
from __future__ import absolute_import
try:
  from enthought.traits.api import *
except:
  from traits.api import *
