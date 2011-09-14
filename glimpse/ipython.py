# A set of IPython helper methods

from glimpse import util
import numpy
np = numpy
from glimpse import core
import os
#import odd

load = util.Load
store = util.Store

def stats(x):
  return dict((k, getattr(x, k)()) for k in ('min', 'max', 'mean', 'std'))

def array3D(x):
  """Reshape array to have three dimensions, collapsing remaining dimensions."""
  return x.reshape((-1,) + x.shape[-2:])

def array2D(x):
  """Reshape array to have two dimensions, collapsing remaining dimensions."""
  return x.reshape((-1, x.shape[-1]))

WITH_PLOT = ('DISPLAY' in os.environ)
if WITH_PLOT:
  import glimpse.util.plot as gplot
  from matplotlib import pyplot
  def hist(x, **args):
    from matplotlib import pylab
    default_args = dict(bins = 100, hold = False)
    return pylab.hist(x.flat, **dict(default_args.items() + args.items()))

  def show(x, **args):
    if hasattr(x, 'show'):
      x.show()
    elif isinstance(x, np.ndarray):
      from matplotlib import pyplot
      pyplot.clf()
      if len(x.shape) == 1:
        pyplot.plot(x, **args)
      elif len(x.shape) == 2:
        gplot.Show3DArray(x, **args)
      else:
        gplot.Show3DArray(x, **args)
      pyplot.draw()
    elif isinstance(x, list) or isinstance(x, tuple):
      if isinstance(x[0], np.ndarray):
        gplot.Show2DArrayList(x, **args)
      else:
        raise ValueError("Can't display list for objects of type: %s" % type(x))
    else:
      raise ValueError("Can't display object of type: %s" % type(x))
