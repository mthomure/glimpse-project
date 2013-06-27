"""Miscellaneous functions for plotting data with Matplotlib."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import math
import numpy as np
import operator
import sys

from glimpse.util.gimage import PowerSpectrum

def InitPlot(use_file_output = False):
  """Initialize matplotlib plotting library, optionally configuring it to
  write plots to disk."""
  import matplotlib
  if use_file_output:
    matplotlib.use("cairo")
  elif matplotlib.get_backend() == "":
    matplotlib.use('TkAgg')
  import matplotlib.pyplot as plt
  return plt

_sp_y = _sp_x = _sp_j = 0
def MakeSubplot(y, x):
  global _sp_y, _sp_x, _sp_j
  _sp_y = y
  _sp_x = x
  _sp_j = 1

def NextSubplot(figure = None):
  from matplotlib import pyplot  # import must be delayed
  global _sp_j
  if figure == None:
    figure = pyplot.gcf()
  figure.add_subplot(_sp_y, _sp_x, _sp_j)
  _sp_j += 1
  return figure.gca()

def Show2dArray(fg, bg = None, mapper = None, annotation = None, title = None,
    axes = None, show = True, colorbar = False, alpha = None, **fg_args):
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
  :param axes: The axes object on which to draw the array contents.
  :param bool show: Whether to show the figure on screen after it is drawn.
  :param bool colorbar: Whether to show the meaning of the colormap as an extra
     subplot.

  Any remaining keyword arguments will be passed to
  :func:`matplotlib.pyplot.imshow`.

  """
  from matplotlib import pyplot  # import must be delayed
  from matplotlib import cm
  if axes == None:
    axes = pyplot.gca()
  if bg == None:
    f_extent = [0, fg.shape[-1], fg.shape[-2], 0]
    f_extent[1] += 1
    f_extent[2] += 1
    b_extent = f_extent
    if alpha == None:
      alpha = 1.0
  else:
    f_extent = (0, fg.shape[-1], fg.shape[-2], 0)
    if mapper != None:
      f_extent = map(mapper, (0, fg.shape[-1], fg.shape[-2], 0))
      f_extent[1] += 1
      f_extent[2] += 1
    b_extent = (0, bg.shape[-1], bg.shape[-2], 0)
    if alpha == None:
      alpha = 0.5
  # Show smallest element first -- that means annotation, fg, bg
  hold = axes.ishold()
  if annotation != None:
    # Show annotation in upper left corner
    k_extent = (0, b_extent[1] * 0.2, b_extent[2] * 0.2, 0)
    axes.imshow(annotation, cmap = cm.gray, extent = k_extent,
        zorder = 3)
  axes.hold(True)
  # Show foreground
  axes.imshow(fg, **dict(fg_args, # cmap = cm.spectral,
      extent = f_extent, zorder = 2, alpha = alpha))
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

#: .. deprecated:: 0.1.0
#:    Use :func:`Show2dArray` instead.
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
  from matplotlib import pyplot  # import must be delayed
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
    figure = None, show = True, title = None, titles = None, **args):
  """Display a list of 2-D arrays using matplotlib.

  :param xs: Input arrays.
  :type xs: iterable of 2D ndarray
  :param annotations: Small images to show with each array visualization.
  :type annotations: list of 2D ndarray
  :param bool normalize: Whether to use the same colormap range for all
     subplots.
  :param bool colorbar: Whether to show the meaning of the colormap as an extra
     subplot (implies *normalize* = True). Implies ``colorbars`` is False.
  :param bool colorbars: Whether to show a different colorbar for each subplot.
  :param int cols: Number of subplot columns to use.
  :param bool center_zero: When normalizing a range that spans zero, make sure
     zero is in the center of the colormap range.
  :param figure: The matplotlib figure into which to plot.
  :param bool show: Whether to show the figure on screen after it is drawn.
  :param str title: String to display above the set of plots.
  :param titles: Title string to include above each plotted array.
  :type titles: list of str
  :param vmin: Minimum value for colormap range.
  :param vmax: Maximum value for colormap range.
  :param mapper: Function to map locations in the foreground to corresponding
     locations in the background. (Required if *bg* is set.)

  Any remaining keyword arguments will be passed to
  :func:`matplotlib.pyplot.imshow` for each 2D array.

  """
  from matplotlib import pyplot  # import must be delayed
  from mpl_toolkits.axes_grid import AxesGrid  # import must be delayed
  if 'vmin' in args and 'vmax' in args:
    vmin = args['vmin']
    vmax = args['vmax']
  elif colorbar or normalize:
    vmin = min([ x.min() for x in xs ])
    vmax = max([ x.max() for x in xs ])
    if center_zero and vmin < 0 and vmax > 0:
      vmax = max(abs(vmin), vmax)
      vmin = -vmax
    args = dict(args, vmin = vmin, vmax = vmax)
  if annotations == None:
    annotations = [None] * len(xs)
  else:
    assert len(annotations) == len(xs), \
        "Got %d arrays, but %d annotations (these numbers should match)" % \
        (len(xs), len(annotations))
    assert 'mapper' in args, "Using annotations requires a 'mapper'."
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
  if title is not None:
    figure.suptitle(title)
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

#: .. deprecated:: 0.1.0
#:    Use :func:`Show2dArrayList` instead.
Show2DArrayList = Show2dArrayList

def Show3dArray(xs, annotations = None, figure = None, **args):
  """Display slices of a 3-D array using matplotlib.

  :param xs: Input arrays.
  :type xs: N-D ndarray, where N > 2.
  :param annotations: Small images to show with each array visualization.
  :type annotations: list of 2D ndarray
  :param figure: The matplotlib figure into which to plot.
  :param bool normalize: Whether to use the same colormap range for all
     subplots.
  :param bool colorbar: Whether to show the meaning of the colormap as an extra
     subplot (implies *normalize* = True). Implies ``colorbars`` is False.
  :param bool colorbars: Whether to show a different colorbar for each subplot.
  :param int cols: Number of subplot columns to use.
  :param bool center_zero: When normalizing a range that spans zero, make sure
     zero is in the center of the colormap range.
  :param figure: The matplotlib figure into which to plot.
  :param bool show: Whether to show the figure on screen after it is drawn.
  :param str title: String to display above the set of plots.
  :param titles: Title string to include above each plotted array.
  :type titles: list of str
  :param vmin: Minimum value for colormap range.
  :param vmax: Maximum value for colormap range.
  :param mapper: Function to map locations in the foreground to corresponding
     locations in the background. (Required if *bg* is set.)

  Any remaining keyword arguments will be passed to
  :func:`matplotlib.pyplot.imshow` for each 2D array slice.

  """
  xs = xs.reshape((-1,) + xs.shape[-2:])
  if annotations != None:
    annotations = annotations.reshape((-1,) + annotations.shape[-2:])
  Show2dArrayList(xs, annotations, figure = figure, **args)

#: .. deprecated:: 0.1.0
#:    Use :func:`Show3dArray` instead.
Show3DArray = Show3dArray

def Show(data, **kwargs):
  """Display a dataset as one or more images.

  :param data: Input data to display. Must be N-dim array-like.
  :type data: N-dim ndarray, or list of 2D ndarray

  """
  if isinstance(data, np.ndarray):
    if data.ndim == 2:
      func = Show2dArray
    else:
      func = Show3dArray
  elif hasattr(data, '__iter__'):
    func = Show2dArrayList
  else:
    raise ValueError("Data must be an array or a list.")
  func(data, **kwargs)

def ShowImagePowerSpectrum(data, width = None, **plot_args):
  """Display the 1-D power spectrum of an image.

  :param data: Image data.
  :type data: 2D ndarray
  :param int width: Effective width of image (with padding) for FFT.

  Any remaining keyword arguments will be passed to
  :func:`matplotlib.pyplot.plot`.

  """
  from matplotlib import pyplot  # import must be delayed
  freqs, power, cnts = PowerSpectrum(data, width)
  pyplot.plot(freqs, power, **plot_args)
  pyplot.yticks([])
  pyplot.xlabel('Cycles per Pixel')
  pyplot.ylabel('Power')

def BarPlot(xs, yerr = None, ax = None, glabels = None, clabels = None,
    colors = None, ecolors = None, gwidth = .7, cwidth = 1., **kwargs):
  """Plot data as bars.

  :param xs: Input data, organized by group and then category (if 2D), or just by
     category (if 1D).
  :type xs: 1D or 2D array-like
  :param yerr: Size of error bars.
  :type yerr: 1D or 2D array-like (must match shape of xs)
  :param ax: Matplotlib axes on which to plot.
  :param glabels: Label for each group (appears on X axis).
  :type glabels: list of str
  :param clabels: Label for each category (appears in legend).
  :type clabels: list of str
  :param colors: Bar color for each category.
  :type colors: list of str
  :param ecolors: Color of error bars for each category. Default is black.
  :type ecolors: list of str
  :param float gwidth: Fraction of available space to use for group of bars.
  :param float cwidth: Fraction of available space to use for each category bar.
  :param dict kwargs: Arguments passed to pyplot.bar() method.
  :returns: Figure into which the data is plotted.

  Note that group labels (glabels) appear along the x-axis, while category
  labels (clabels) appear in the legend.

  """
  import matplotlib.pyplot as plt
  xs = np.asarray(xs)
  if yerr is not None:
    # This must come before xs is reshaped below
    yerr = np.array(yerr)
    if xs.shape != yerr.shape:
      raise ValueError("Data array and errors array must have same shape")
    yerr = np.atleast_2d(yerr).T  # rotate yerr to be indexed first by category
  if xs.ndim > 2:
    raise ValueError("Data array must be 1D or 2D")
  xs = np.atleast_2d(xs).T  # rotate xs to be indexed by category, then group
  C, G = xs.shape[:2]
  gwidth = min(1, max(0, gwidth))
  total_gwidth = 1  # available space for each group
  used_gwidth = total_gwidth * gwidth  # space actually used for each group
  goffset = (total_gwidth - used_gwidth) / 2  # offset of drawable region from
                                              # edge of group
  gedges = np.arange(G) * total_gwidth + goffset  # edges of drawable region for
                                                  # each group
  cwidth = min(1, max(0, cwidth))
  total_cwidth = used_gwidth / C  # available space for each category bar
  used_cwidth = total_cwidth * cwidth  # space actually used for each bar
  coffset = (total_cwidth - used_cwidth) / 2  # offset of drawable region from
                                              # edge of category
  gind = np.arange(G)
  cind = np.arange(C)
  if colors == None:
    colors = [None] * C
  else:
    colors = list(colors) + [None] * C  # potentially too long, but that's ok
  if ecolors == None:
    ecolors = "k" * C
  else:
    ecolors = list(ecolors) + ['k'] * C
  if yerr == None:
    yerr = [None] * C
  if ax == None:
    ax = plt.gca()
  if clabels == None:
    clabels = [''] * C
  else:
    clabels = list(clabels) + [''] * C
  # For each category, create a bar in all groups.
  crects = [ ax.bar(
      (gind * total_gwidth + goffset) + (c * total_cwidth + coffset),  # min edges
      xs[c],                # height of column
      used_cwidth,          # column width
      color = colors[c],    # column color
      yerr = yerr[c],       # error
      ecolor = ecolors[c],  # color of error bar
      label = clabels[c],
      **kwargs
      ) for c in cind ]
  if glabels == None:
    # Disable x-ticks.
    ax.set_xticks([])
  else:
    # Show x-tick labels.
    ax.set_xticks((gind + .5) * total_gwidth)
    ax.set_xticklabels(glabels)
    ax.tick_params(axis = 'x', direction = 'out')
    ax.xaxis.tick_bottom()  # hide top tick marks
  ax.yaxis.tick_left()  # hide right tick marks
  if plt.isinteractive():
    ax.figure.show()
  return ax

def SimpleBarPlot(xs, ys, label = None, ax = None, **kw):
  """Plot a dataset as vertical bars."""
  import matplotlib.pyplot as plt
  if ax is None:
    ax = plt.gca()
  rects = ax.bar(xs, ys, **kw)
  if label is not None:
    rects[0].set_label(label)
  plt.draw_if_interactive()
  return rects

def Scatter3d(xs, ys, zs=None, weights=None, xlabel=None, ylabel=None, zlabel=None, **kw):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  ax = plt.gcf().add_subplot(111, projection='3d')
  if not any(x in kw for x in ('lw', 'linewidth', 'linewidths')):
    kw['lw'] = .3
  if weights is not None:
    kw['c'] = weights
  if zs is None:
    zs = [0] * len(xs)
  ax.scatter(xs, ys, zs, **kw)
  if xlabel:
    plt.xlabel(xlabel)
  if ylabel:
    plt.ylabel(ylabel)
  if zlabel:
    ax.zaxis.label.set_text(zlabel)

def Scatter3d_Mayavi(xs, ys, zs, weights=None, weight_label=None, xlabel='x',
    ylabel='y', zlabel='z', axes=True, **points3d_kw):
  from mayavi import mlab
  # XXX error when max() returns zero
  def normalize(vs):
    a = vs.min()
    b = vs.max()
    if a != b:
      vs = vs / float(b - a)
    return vs
  xs = normalize(xs)
  ys = normalize(ys)
  zs = normalize(zs)
  if 'scale_mode' not in points3d_kw:
    points3d_kw['scale_mode'] = 'none'
  if 'scale_factor' not in points3d_kw:
    points3d_kw['scale_factor'] = .03
  if weights is None:
    mlab.points3d(xs, ys, zs, **points3d_kw)
  else:
    mlab.points3d(xs, ys, zs, weights, **points3d_kw)
  if weight_label:
    mlab.scalarbar(title=weight_label)
  if axes:
    extent = 1.2
    scale = 0.05
    tube_radius = 0.01
    color = (1,1,1)  # white
    mlab.plot3d([0,extent], [0,0], [0,0], tube_radius=tube_radius, color=color)
    mlab.plot3d([0,0], [0,extent], [0,0], tube_radius=tube_radius, color=color)
    mlab.plot3d([0,0], [0,0], [0,extent], tube_radius=tube_radius, color=color)
    if xlabel:
      mlab.text3d(extent, 0, 0, xlabel, scale=scale, color=color)
    if ylabel:
      mlab.text3d(0, extent, 0, ylabel, scale=scale, color=color)
    if zlabel:
      mlab.text3d(0, 0, extent, zlabel, scale=scale, color=color)

def PlotHistogram2d(x, y, bins = 10, range=None, weights=None, cmin=None,
    cmax=None, **kwargs):
  # Copied from https://github.com/matplotlib/matplotlib/pull/805
  import matplotlib.pyplot as plt
  # xrange becomes range after 2to3
  bin_range = range
  range = __builtins__["range"]
  h,xedges,yedges = np.histogram2d(x, y, bins=bins, range=bin_range,
      normed=False, weights=weights)
  if 'origin' not in kwargs: kwargs['origin']='lower'
  if 'extent' not in kwargs: kwargs['extent']=[xedges[0], xedges[-1], yedges[0],
      yedges[-1]]
  if 'interpolation' not in kwargs: kwargs['interpolation']='nearest'
  if 'aspect' not in kwargs: kwargs['aspect']='auto'
  if cmin is not None: h[h<cmin]=None
  if cmax is not None: h[h>cmax]=None
  im = plt.imshow(h.T, **kwargs)
  return h,xedges,yedges,im

def BarPlot2(*datasets):
  import matplotlib.pyplot as plt
  from .util import ys_mean
  padding = .1
  N = len(datasets)
  width = .8 / N
  xs = np.array([ np.unique(d[:,0]) for d in datasets ])
  left_edge = np.array(len(xs))
  colors = 'rgb'
  for idx,data in enumerate(datasets):
    plt.bar(left_edge + padding + idx * width, ys_mean(data), color=colors[idx % len(colors)])
