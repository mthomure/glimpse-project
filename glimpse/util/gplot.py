"""Miscellaneous functions for plotting data with Matplotlib."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import math
import matplotlib
from matplotlib import cm
import numpy as np
import operator
import sys

from . import gimage
from . import gio
from . import misc

def InitPlot(use_file_output = False):
  """Initialize matplotlib plotting library, optionally configuring it to
  write plots to disk."""
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
  axes.imshow(fg, **misc.MergeDict(fg_args, # cmap = cm.spectral,
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
    args = misc.MergeDict(args, vmin = vmin, vmax = vmax)
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
  elif misc.IsIterable(data):
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
  freqs, power, cnts = gimage.PowerSpectrum(data, width)
  pyplot.plot(freqs, power, **plot_args)
  pyplot.yticks([])
  pyplot.xlabel('Cycles per Pixel')
  pyplot.ylabel('Power')

def BarPlot(xs, yerr = None, fig = None, glabels = None, clabels = None,
    colors = None, ecolors = None, gwidth = .7, cwidth = 1., show = True,
    **kwargs):
  """Plot data as bars.

  :param xs: Input data, organized by group and then category (if 2D), or just by
     category (if 1D).
  :type xs: 1D or 2D array-like
  :param yerr: Size of error bars.
  :type yerr: 1D or 2D array-like (must match shape of xs)
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
  :param bool show: Whether to show the figure after plotting.
  :param dict kwargs: Arguments passed to pyplot.bar() method.
  :returns: Figure into which the data is plotted.

  """
  import matplotlib.pyplot as plt
  xs = np.asarray(xs)
  if xs.ndim > 2:
    raise ValueError("Data array must be 1D or 2D")
  if yerr != None:
    yerr = np.asarray(yerr)
    if xs.shape != yerr.shape:
      raise ValueError("Data array and errors array must have same shape")
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
  if ecolors == None:
    ecolors = "k" * C
  if yerr == None:
    yerr = [None] * C
  else:
    yerr = np.atleast_2d(yerr).T  # rotate yerr to be indexed first by category
  if fig == None:
    fig = plt.figure()
  else:
    fig.clf()  # clear the figure
  ax = fig.add_subplot(111)
  clabels = clabels or [''] * C
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
  if show:
    fig.show()
  return ax

##### Command-Line Interface #######################

def _Summarize2D(data):
  """Summarize a 2D dataset with repeated observations for X-values.

  :param data: (X,Y) value pairs, not necessarily sorted by X-value.
  :type data: 2D-ndarray
  :returns: Mean and standard error (STDDEV / SQRT(COUNT)) for each unique X
     value, where each row has the form (X, MEAN[Y], STDERR[Y]).
  :rtype: 2D-ndarray

  """
  if data.size == 0:
    return np.array([]).reshape(0, 3)
  keys = data[:, 0]
  results = []
  for key in np.unique(keys):
    subset = data[ keys == key ][:, 1]
    mean = subset.mean()
    err = subset.std() / math.sqrt(len(subset))  # compute standard error
    results.append([ key, mean, err ])
  return np.array(results)

class Loader(object):
  """Loads data from files."""

  #: File encoding used by Read* methods.
  input_encoding = gio.ENCODING_PICKLE

  def Read(self, fname):
    try:
      return list(gio.LoadAll(fname, self.input_encoding))[0]
    except Exception:
      raise Exception("Failed to load dataset, maybe wrong input type (see -i "
          "option)?")

  def ReadData1D(self, fname):
    """Read a file with a single column per line."""
    data = np.asarray(self.Read(fname))
    if data.ndim != 2 or data.shape[-1] != 1:
      raise Exception("Dataset has wrong shape: expected array with shape "
          "(N, 1), but got %s for file %s" % (data.shape, fname))
    return data.reshape(data.shape[0])

  def ReadData2D(self, fname):
    """Read a file with multiple columns per line."""
    data = self.Read(fname)
    if data.size == 0:
      data = data.reshape(0, 2)
    if data.shape[1] != 2:
      raise Exception("Dataset has wrong shape: expected array with shape "
          "(N, 2), but got %s for file %s" % (data.shape, fname,))
    return data

class Plotter(object):

  TYPE_BAR = 1
  TYPE_HIST = 2
  TYPE_IMAGE = 3
  TYPE_LINE = 4
  TYPE_SCATTER = 5

  @staticmethod
  def _GetPlotFunc(plot_type):
    funcs = {
        Plotter.TYPE_BAR : Plotter.BarPlot,
        Plotter.TYPE_HIST : Plotter.HistPlot,
        Plotter.TYPE_IMAGE : Plotter.ImagePlot,
        Plotter.TYPE_LINE : Plotter.LinePlot,
        Plotter.TYPE_SCATTER : Plotter.ScatterPlot,
    }
    func = funcs.get(plot_type, None)
    if func == None:
      raise ValueError("Unknown plot type: %s" % plot_type)
    return func

  #: Limits for y- and x-axis (4-tuple of int).
  axis = None
  #: Line colors, or category colors for bar plot (list of str).
  colors = None
  #: Category labels for bar plots (list of str).
  bar_clabels = None
  #: Number of categories per group for bar plot (int).
  bar_group_size = None
  #: Whether to plot error bars (bool).
  error_bars = False
  #: Number of bins to use for histogram plot (int).
  hist_nbins = 100
  #: Line labels, or group labels for bar plot (list of str).
  labels = None
  #: Line styles (list of str).
  linestyles = None
  loader = Loader()
  #: Path of image file to which plot is saved (str). See PostProcess().
  ofname = None
  #: Type of plot to generate (one of Plotter.TYPE_*).
  plot_type = TYPE_LINE
  #: Command to interpret after plot is generated (str).
  post_command = None
  #: Plot title (str).
  title = None
  #: Label of x-axis (str).
  xlabel = None
  #: Label of y-axis (str).
  ylabel = None

  def PreProcess(self):
    self.plt = InitPlot(self.ofname != None)

  def BarPlot(self, data_sets):
    """Plot a collection of 1D datasets as groups of bars."""
    xs = np.asarray(data_sets)
    if self.error_bars:
      yerr = xs.std(-1) / math.sqrt(xs.shape[-1])  # standard error
      xs = xs.mean(-1)
    else:
      yerr = None
    kwargs = dict(linewidth = 0)
    BarPlot(xs, yerr = yerr, colors = self.colors, glabels = self.labels,
        clabels = self.bar_clabels, show = False, **kwargs)

  def HistPlot(self, data_sets):
    """Plot a collection of 1D datasets as overlapping histograms."""
    N = len(data_sets)
    keywords = ('color', 'label', 'linestyle')
    opts = [ getattr(self, kw, None) or []
        for kw in ('colors', 'labels', 'linestyles') ]
    for idx in range(N):
      data = data_sets[idx]
      kwargs = dict( (kw, opt[idx]) for kw, opt in zip(keywords, opts)
          if idx < len(opt) )
      self.plt.hist(data.flat, self.hist_nbins, **kwargs)

  def ImagePlot(self, data):
    """Plot a single dataset as images.

    :param data: Input data. If data has more than 2 dimensions, it is treated as
       list of 2D arrays, with each plotted in a separate subplot.
    :type data: ND array, with N>1

    """
    Show2dArray(data)

  def LinePlot(self, data_sets):
    """Plot a collection of 2D datasets as lines."""
    N = len(data_sets)
    keywords = ('color', 'label', 'linestyle')
    opts = [ getattr(self, kw, None) or []
        for kw in ('colors', 'labels', 'linestyles') ]
    for idx in range(N):
      data = data_sets[idx]
      kwargs = dict( (kw, opt[idx]) for kw, opt in zip(keywords, opts)
          if idx < len(opt) )
      if self.error_bars:
        xs, ymean, yerr = _Summarize2D(data).T
        if yerr.size == 0:
          yerr = None
        self.plt.errorbar(xs, ymean, yerr = yerr, **kwargs)
      else:
        data.sort(0)
        data = data.T
        self.plt.plot(data[0], data[1], **kwargs)

  def ScatterPlot(self, data_sets):
    """Plot a collection of 2D datasets as disconnected points."""
    N = len(data_sets)
    keywords = ('color', 'label', 'linestyle')
    opts = [ getattr(self, kw, None) or []
        for kw in ('colors', 'labels', 'linestyles') ]
    for idx in range(N):
      data = data_sets[idx]
      kwargs = dict( (kw, opt[idx]) for kw, opt in zip(keywords, opts)
          if idx < len(opt) )
      data = data.T
      data.sort(0)
      self.plt.scatter(data[0], data[1], **kwargs)

  def PlotFromFile(self, *fnames, **kwargs):
    """Plot a collection of datasets read from disk.

    :param fnames: Path to datasets.
    :type fnames: list of str
    :param bool post_process: Whether to call :meth:`PostProcess` (default is
       True).

    If needed, :meth:`PreProcess` is called automatically.

    """
    post_process = kwargs.get('post_process', True)
    if self.plot_type == Plotter.TYPE_BAR:
      # Bar plots use 1D datasets
      data_sets = map(self.loader.ReadData1D, fnames)
      data_sets = np.asarray(data_sets)
      # Reshape data_sets from 2D (given by file, line) to 3D (given by file
      # group, file category, line).
      gsize = self.bar_group_size or len(data_sets)
      if data_sets.shape[0] % gsize != 0:
        raise ValueError("Number of datasets must be multiple of group size")
      data_sets = data_sets.reshape(-1, gsize, data_sets.shape[-1])
    elif self.plot_type == Plotter.TYPE_HIST:
      # Histograms use 1D datasets
      data_sets = map(self.loader.ReadData1D, fnames)
    elif self.plot_type in (Plotter.TYPE_HIST, Plotter.TYPE_IMAGE,
        Plotter.TYPE_LINE, Plotter.TYPE_SCATTER):
      # All other plots use 2D datasets
      data_sets = map(self.loader.ReadData2D, fnames)
    else:
      raise ValueError("Unknown plot type: %s" % self.plot_type)
    plot_func = Plotter._GetPlotFunc(self.plot_type)
    if not hasattr(self, 'plt'):
      self.PreProcess()
    ret = plot_func(self, data_sets)
    if post_process:
      self.PostProcess()

  def PostProcess(self):
    """Apply desired post-processing to generated plot, and display the result.

    The plot is displayed to the screen, unless :attr:`ofname` is set. In the
    latter case, the plot is saved to disk.

    """
    if not self.plt:
      raise Exception("PostProcess called before PreProcess")
    plt = self.plt
    if self.axis:
      plt.axis(self.axis)
    if self.title:
      plt.title(self.title)
    if self.xlabel:
      plt.axes().set_xlabel(self.xlabel)
    if self.ylabel:
      plt.axes().set_ylabel(self.ylabel)
    if self.post_command:
      plot = plt
      eval(self.post_command, globals(), locals())
    if self.labels:
      plt.rcParams['legend.loc'] = 'best'
      plt.legend()
    if self.ofname:
      plt.savefig(self.ofname)
    else:
      plt.show()

def _CliMain():
  p = Plotter()
  opts, args = misc.GetOptions('a:bc:C:eg:Hi:Il:L:o:s:St:x:y:')
  # Parse arguments needed early
  for opt, arg in opts:
    if opt == '-a':
      p.axis = map(float, arg.split(","))
    elif opt == '-b':
      p.plot_type = Plotter.TYPE_BAR
    elif opt == '-c':
      p.colors = arg.split(",")
    elif opt == '-C':
      p.post_command = arg
    elif opt == '-e':
      p.error_bars = True
    elif opt == '-g':
      p.bar_group_size = int(arg)
    elif opt == '-i':
      p.loader.input_encoding = arg
    elif opt == '-I':
      p.plot_type = Plotter.TYPE_IMAGE
    elif opt == '-H':
      p.plot_type = Plotter.TYPE_HIST
    elif opt == '-l':
      p.labels = arg.split(",")
    elif opt == '-L':
      p.bar_clabels = arg.split(",")
    elif opt == '-o':
      p.ofname = arg
    elif opt == '-s':
      p.linestyles = arg.split(",")
    elif opt == '-S':
      p.plot_type = Plotter.TYPE_SCATTER
    elif opt == '-t':
      p.title = arg
    elif opt == '-x':
      p.xlabel = arg
    elif opt == '-y':
      p.ylabel = arg
  for i in range(len(args)):
    if args[i] == "-":
      args[i] = sys.stdin
  if len(args) < 1:
    args = [ sys.stdin ]
  p.PlotFromFile(*args)

def main():
  try:
    _CliMain()
  except misc.UsageException, e:
    if e.msg:
      print >>sys.stderr, e.msg
    misc.Usage(
      "[options] [DATA ...]\n" + \
      "options:\n"
      "  -a X0,X1,Y0,Y1  Set range of X and Y axes to [X0,X1] and [Y0,Y1].\n"
      "  -c COLORS       Specify comma-separated line colors, or group colors "
      "for bar\n"
      "                  plot. Ex: \"r,g,b\".\n"
      "  -C COMMAND      Specify a command to evaluate after plotting the data."
      "\n"
      "  -e              Plot 2D datasets with repeated X values by showing"
      " error bars\n"
      "                  as plus or minus the standard error (std / sqrt(#obs))"
      ".\n"
      "  -g SIZE         Specify group size for bar plot.\n"
      "  -H              Plot histogram of 1D datasets.\n"
      "  -i TYPE         Set input encoding type [one of:"
      " %s, default: %s].\n" % (", ".join(gio.INPUT_ENCODINGS),
          gio.ENCODING_PICKLE) + \
      "  -I              Plot ND datasets as images.\n"
      "  -l LABELS       Specify comma-separated line names, or group names for "
      "bar\n"
      "                  plot.\n"
      "  -L LABELS       Specify category labels for bar plot.\n"
      "  -o FNAME        Write plot to image file FNAME.\n"
      "  -s STYLES       Specify comma-separated line styles. Ex: solid, '-',"
      " \n"
      "                  dashed, '--'.\n"
      "  -S              Show 2D datasets as a scatterplot.\n"
      "  -t TITLE        Set chart title.\n"
      "  -x LABEL        Set label on x-axis.\n"
      "  -y LABEL        Set label on y-axis.\n"
      "For more option values, see:\n"
      "  http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib"
      ".pyplot.plot\n"
      "To use log scale on the X-axis:\n"
      "  -C 'plot.gca().set_xscale(\"log\")'"
    )

if __name__ == '__main__':
  main()
