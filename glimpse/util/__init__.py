
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Import the most common methods into the glimpse.util namespace
#

from garray import *
from gimage import *
from gio import *
from misc import *
from gos import *


import time
class Timer:
  def __init__(self, name, enabled = True):
    self.start = None
    self.name = name
    self.enabled = enabled
  def __enter__(self):
    self.start = time.time()
  def __exit__(self, type, value, traceback):
    stop = time.time()
    if self.enabled:
      print "TIME(%s): %.4f secs" % (self.name, stop - self.start)


# This method is not in util.gplot because we generally want to use it before
# importing matplotlib (and thus before importing util.gplot).
_MATPLOTLIB_IMPORT_FLAG = False
def InitPlot(use_file_output = False):
  """Initialize matplotlib plotting library, optionally configuring it to
  write plots to disk."""
  global _MATPLOTLIB_IMPORT_FLAG
  if not _MATPLOTLIB_IMPORT_FLAG:
    import matplotlib
    _MATPLOTLIB_IMPORT_FLAG = True
    # Set the backend, if it's not been done already.
    if matplotlib.get_backend() == "":
      matplotlib.use('TkAgg')
    if use_file_output:
      matplotlib.use("cairo")
    import matplotlib.pyplot as plot
    import warnings
#    with warnings.catch_warnings():
#      warnings.filterwarnings("ignore", category = DeprecationWarning)
    plot.clf()
    return plot
  import matplotlib.pyplot as plot
  return plot

