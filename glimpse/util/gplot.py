"""Miscellaneous functions for plotting data with Matplotlib."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

from glimpse.util import gimage, misc
import math
# Following imports assume matplotlib environment has been initialized.
try:
  from matplotlib import cm
  from matplotlib import pyplot
  from mpl_toolkits.axes_grid import AxesGrid
except RuntimeError:
  import logging
  logging.warn("Unable to import matplotlib. Plotting may fail.")
import numpy as np
import operator

_sp_y = _sp_x = _sp_j = 0
def MakeSubplot(y, x):
  global _sp_y, _sp_x, _sp_j
  _sp_y = y
  _sp_x = x
  _sp_j = 1

def NextSubplot(figure = None):
  global _sp_j
  if figure == None:
    figure = pyplot.gcf()
  figure.add_subplot(_sp_y, _sp_x, _sp_j)
  _sp_j += 1
  return figure.gca()

def Show2dArray(fg, bg = None, mapper = None, annotation = None, title = None,
    axes = None, show = True, colorbar = False, **fg_args):
  """Show a 2-D array using matplotlib.

  :param fg: Foreground array to display, shown with a spectral colormap.
  :type fg: 2D ndarray.
  :param bg: Optional background array, shown as grayscale.
  :type bg: 2D ndarray
  :param mapper: Function to map locations in the foreground to corresponding
     locations in the background. (Required if *bg* is set.)
  :param annotation: Small array to display in top-left corner of plot.
  :type annotation: 2D ndarray
  :param str title: String to display above plot.

  """
  if axes == None:
    axes = pyplot.gca()
  if bg == None:
    f_extent = [0, fg.shape[-1], fg.shape[-2], 0]
    f_extent[1] += 1
    f_extent[2] += 1
    b_extent = f_extent
    falpha = 1
  else:
    assert mapper != None
    f_extent = map(mapper, (0, fg.shape[-1], fg.shape[-2], 0))
    f_extent[1] += 1
    f_extent[2] += 1
    falpha = 0.9
    b_extent = (0, bg.shape[-1], bg.shape[-2], 0)
  # Show smallest element first -- that means annotation, fg, bg
  hold = axes.ishold()
  if annotation != None:
    # Show annotation in upper left corner
    k_extent = (0, b_extent[1] * 0.2, b_extent[2] * 0.2, 0)
    axes.imshow(annotation, cmap = cm.gray, extent = k_extent,
        zorder = 3)
  axes.hold(True)
  # Show foreground
  axes.imshow(fg, **misc.MergeDict(fg_args, # cmap = cm.spectral,
      extent = f_extent, zorder = 2, alpha = falpha))
  if colorbar:
    axes.figure.colorbar(axes.images[-1], ax = axes)
  if bg != None:
    # Show base image in the background
    axes.imshow(bg, cmap = cm.gray, extent = b_extent, zorder = 1)
  if title != None:
    axes.set_title(title)
  # Leave the border around the image, but don't show any tick marks.
  axes.set_yticks([])
  axes.set_xticks([])
  axes.hold(hold)
  if hasattr(axes.figure, 'show') and show:
    axes.figure.show()

#: .. deprecated:: 0.1
#:    Use Show2dArray instead.
Show2DArray = Show2dArray

def ShowListWithCallback(xs, cb, cols = None, figure = None, nsubplots = None,
    colorbar = False):
  """Show each slice of a 3-D array using a callback function.

  :param iterable xs: Values to pass to callback function. For example, this
     could be a 3-D array (which is a list of 2-D arrays), or a 1-D array of
     indices.
  :param callable cb: Function taking an axes object and the current dataset.
  :param int nsubplots: Number of total subplots. This could be more than the
     number of elements in *xs*.

  """
  if figure == None:
    figure = pyplot.gcf()
  axes = figure.gca()
  figure.clf()  # necessary to avoid showing large axes in background
  max_plots = 64
  if len(xs) > max_plots:
    xs = xs[:max_plots]
  if nsubplots == None:
    nsubplots = len(xs)
  if cols == None:
    if nsubplots <= 12:
      cols = 4
    else:
      cols = 8
    cols_was_none = True
  else:
    cols_was_none = True
  rows = math.ceil(nsubplots / float(cols))
  if cols_was_none and rows == 1:
    cols = nsubplots
  MakeSubplot(rows, cols)
  for x in xs:
    NextSubplot(figure)
    cb(x)

def Show2dArrayList(xs, annotations = None, normalize = True, colorbar = False,
    colorbars = False, cols = None, center_zero = True, axes = None,
    figure = None, show = True, titles = None, **args):
  """Display a list of 2-D arrays using matplotlib.

  :param annotations: Small images to show with each array visualization.
  :type annotations: list of 2D ndarray
  :param bool normalize: Whether to use the same colormap range for all
     subplots.
  :param bool colorbar: Whether to show the meaning of the colormap as an extra
     subplot (implies *normalize* = True).
  :param bool colorbars: Whether to show a different colorbar for each subplot.
  :param int cols: Number of subplot columns to use.
  :param bool center_zero: When normalizing a range that spans zero, make sure
     zero is in the center of the colormap range.
  :param titles: Title string to include above each plotted array.
  :type titles: list of str

  """
  if 'vmin' in args and 'vmax' in args:
    vmin = args['vmin']
    vmax = args['vmax']
  elif colorbar or normalize:
    vmin = min([ x.min() for x in xs ])
    vmax = max([ x.max() for x in xs ])
    if center_zero and vmin < 0 and vmax > 0:
      vmax = max(abs(vmin), vmax)
      vmin = -vmax
    args = misc.MergeDict(args, vmin = vmin, vmax = vmax)
  if annotations == None:
    annotations = [None] * len(xs)
  else:
    assert len(annotations) == len(xs), \
        "Got %d arrays, but %d annotations (these numbers should match)" % \
        (len(xs), len(annotations))
  if titles == None:
    titles = [""] * len(xs)
  else:
    assert len(titles) == len(xs), \
        "Got %d arrays, but %d title strings (these numbers should match)" % \
        (len(xs), len(titles))
 # Compute rows & cols
  max_plots = 64
  if len(xs) > max_plots:
    xs = xs[:max_plots]
  if cols == None:
    if len(xs) <= 16:
      cols = min(4, len(xs))
    else:
      cols = min(8, len(xs))
  rows = int(math.ceil(len(xs) / float(cols)))
  # Create the image grid
  if figure == None:
    figure = pyplot.gcf()
  figure.clf()
  if colorbar:
    grid_args = dict(cbar_location = "right", cbar_mode = "single")
  elif colorbars:
    grid_args = dict(cbar_location = "right", cbar_mode = "each",
        cbar_pad = "2%")
  else:
    grid_args = {}
  grid = AxesGrid(figure, 111,
    nrows_ncols = (rows, cols),
    axes_pad = 0.5,
    share_all = True,
    label_mode = "L",   # XXX value can be "L" or "1" -- what is this?
    **grid_args
  )
  # Add all subplots
  for i in range(len(xs)):
    Show2dArray(xs[i], annotation = annotations[i], axes = grid[i],
        show = False, title = titles[i], **args)
  # Add the colorbar(s)
  if colorbar:
    img = grid[0].images[-1]
    grid.cbar_axes[0].colorbar(img)
    for cax in grid.cbar_axes:
      cax.toggle_label(True)
  elif colorbars:
    for i in range(len(xs)):
      img = grid[i].images[-1]
      ca = grid.cbar_axes[i]
      ca.colorbar(img)
      ca.toggle_label(True)
  if hasattr(figure, 'show') and show:
    figure.show()

#: .. deprecated:: 0.1
#:    Use Show2dArrayList instead.
Show2DArrayList = Show2dArrayList

def Show3dArray(xs, annotations = None, figure = None, **args):
  """Display slices of a 3-D array using matplotlib.

  :param annotations: Small images to show with each array visualization.
  :type annotations: list of 2D ndarray
  :param figure: The matplotlib figure into which to plot.
  :param args: Arguments to pass to :func:`Show2dArrayList`.

  """
  xs = xs.reshape((-1,) + xs.shape[-2:])
  if annotations != None:
    annotations = annotations.reshape((-1,) + annotations.shape[-2:])
  Show2dArrayList(xs, annotations, figure = figure, **args)

#: .. deprecated:: 0.1
#:    Use Show3dArray instead.
Show3DArray = Show3dArray

def ShowImagePowerSpectrum(data, width = None, **plot_args):
  """Display the 1-D power spectrum of an image.

  :param data: Image data.
  :type data: 2D ndarray
  :param int width: Effective width of image (with padding) for FFT.
  :param dict plot_args: Optional arguments passed to matplotlib :func:`plot`
     command.

  """
  freqs, power, cnts = gimage.PowerSpectrum(data, width)
  pyplot.plot(freqs, power, **plot_args)
  pyplot.yticks([])
  pyplot.xlabel('Cycles per Pixel')
  pyplot.ylabel('Power')
