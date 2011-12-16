# A set of IPython helper methods

from glimpse import util
import numpy
np = numpy
import os
#import odd

load = util.Load
store = util.Store

class stats(object):
  def __init__(self, x):
    self.min = x.min()
    self.max = x.max()
    self.mean = x.mean()
    self.std = x.std()
  def __str__(self):
    return "min = %s, max = %s, mean = %s, std = %s" % (self.min, self.max, self.mean, self.std)
  def __repr__(self):
    return str(self)

def array3D(x):
  """Reshape array to have three dimensions, collapsing remaining dimensions."""
  return x.reshape((-1,) + x.shape[-2:])

def array2D(x):
  """Reshape array to have two dimensions, collapsing remaining dimensions."""
  return x.reshape((-1, x.shape[-1]))

WITH_PLOT = ('DISPLAY' in os.environ)
if WITH_PLOT:

  from glimpse.util import gplot
  from matplotlib import pyplot

  def hist(x, **args):
    from matplotlib import pylab
    default_args = dict(bins = 100, hold = False)
    if isinstance(x, numpy.ndarray) and len(x.shape) > 1:
      x = x.flat
    return pylab.hist(x, **dict(default_args.items() + args.items()))

  def show(x, **args):
    if hasattr(x, 'show'):
      x.show()
    elif isinstance(x, np.ndarray):
      from matplotlib import pyplot
      if len(x.shape) == 1:
        pyplot.plot(x, **args)
      elif len(x.shape) == 2:
        gplot.Show3DArray(x, **args)
      else:
        gplot.Show3DArray(x, **args)
      pyplot.draw()
    elif isinstance(x, list) or isinstance(x, tuple):
      from matplotlib import pyplot
      if isinstance(x[0], np.ndarray):
        gplot.Show2DArrayList(x, **args)
        pyplot.draw()
      else:
        raise ValueError("Can't display list for objects of type: %s" % type(x))
    else:
      raise ValueError("Can't display object of type: %s" % type(x))
