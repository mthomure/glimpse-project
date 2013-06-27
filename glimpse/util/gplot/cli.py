# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import sys

from glimpse.util.option import *
from .util import *
from . import gplot

def ReadData1D(fname):
  """Read a file with a single column per line."""
  data = np.loadtxt(fname)
  if data.ndim != 2 or data.shape[-1] != 1:
    raise Exception("Dataset has wrong shape: expected array with shape "
        "(N, 1), but got %s for file %s" % (data.shape, fname))
  return data.reshape(data.shape[0])

def ReadData2D(fname):
  """Read a file with multiple columns per line."""
  data = np.loadtxt(fname)
  if data.size == 0:
    data = data.reshape(0, 2)
  if data.shape[1] != 2:
    raise Exception("Dataset has wrong shape: expected array with shape "
        "(N, 2), but got %s for file %s" % (data.shape, fname,))
  return data

def MakeOptions():
  return OptionRoot(
    Option('axis', flag=('a:','axis='), doc="Set axis limits as 'X0,X1,Y0,X1'"),
    Option('padding', flag='padding=', doc="Set the padding around the plot as 'left,bottom,right,top'"),
    Option('plot_type_bar', flag='b', doc="Plot data as vertical bars"),
    Option('color', flag=('c:','color='), doc="Specify comma-separated line colors, or group colors for bar plot. Ex: 'r,g,b'"),
    Option('command', flag=('C:','command='), doc="Specify a command to evaluate after plotting the data"),
    Option('error_bars', flag='e', doc="Plot 2D datasets with repeated X values by showing error bars as plus or minus the standard error (std / sqrt(#obs))"),
    Option('bar_group_size', flag='g:', doc="Specify group size for bar plot (default is 1)", default=0),
    Option('help', flag=('h', 'help'), doc="Print this help message and exit"),
    Option('plot_type_hist', flag='H', doc="Plot histogram of 1D datasets"),
    Option('plot_type_image', flag='i', doc="Plot ND datasets as images"),
    Option('label', flag=('l:','label='), doc="Specify comma-separated line names, or category labels for bar plot"),
    Option('bar_glabel', flag='L:', doc="Specify group labels for bar plot"),
    Option('ofname', flag=('o:','output='), doc="Write plot to image file FNAME"),
    Option('linestyle', flag=('s:','linestyle='), doc="Specify comma-separated line styles. Ex: solid, '-', dashed, '--'"),
    Option('plot_type_scatter', flag='S', doc="Show 2D datasets as a scatterplot"),
    Option('title', flag=('t:','title='), doc="Set label on x-axis"),
    Option('xlabel', flag=('x:', 'xlabel='), doc="Set label on x-axis"),
    Option('ylabel', flag=('y:', 'ylabel='), doc="Set label on y-axis"),
    Option('logx', flag='logx', doc="Use logarithmic scaling for x-axis"),
    Option('logy', flag='logy', doc="Use logarithmic scaling for y-axis"),
    )

def get_attrs(idx, *names, **kw):
  result = dict()
  for n in names:
    d = kw.get(n)
    # ignore empty list, or list with single empty string
    if d and (len(d) > 1 or d[0] != '') and idx < len(d):
      result[n] = d[idx]
  return result

def BarPlot(data_sets, **kw):
  """Plot a collection of 1D datasets as groups of bars."""
  import matplotlib.pyplot as plt
  xs = np.asarray(data_sets)
  if opts.error_bars:
    yerr = xs.std(-1) / math.sqrt(xs.shape[-1])  # standard error
    xs = xs.mean(-1)
  else:
    yerr = None
  if 'label' in kw:
    kw['clabel'] = kw['label']
    del kw['label']
  gplot.BarPlot(xs, yerr=yerr, show=False,
      linewidth=0,  # remove borders from bars
      **kw)

def HistPlot(data_sets, **kw):
  """Plot a collection of 1D datasets as overlapping histograms."""
  import matplotlib.pyplot as plt
  nbins = kw.get('nbins', 100)
  for idx,data in enumerate(data_sets):
    plt.hist(data.flat, nbins, **get_attrs(idx, 'color', 'label', 'linestyle',
        **kw))

def ImagePlot(data):
  """Plot a single dataset as images.

  :param data: Input data. If data has more than 2 dimensions, it is treated as
     list of 2D arrays, with each plotted in a separate subplot.
  :type data: ND array, with N>1

  """
  Show2dArray(data)

def LinePlot(data_sets, error_bars=False, **kw):
  """Plot a collection of 2D datasets as lines."""
  import matplotlib.pyplot as plt
  for idx,data in enumerate(data_sets):
    kw_ = get_attrs(idx, 'color', 'label', 'linestyle', **kw)
    if error_bars:
      yerr = ys_stderr(data)
      if len(yerr) == 0:
        yerr = None
      plt.errorbar(xs(data), ys_mean(data), yerr=yerr, **kw_)
    else:
      data.sort(0)
      data = data.T
      plt.plot(data[0], data[1], **kw_)

def ScatterPlot(data_sets, **kw):
  """Plot a collection of 2D datasets as disconnected points."""
  import matplotlib.pyplot as plt
  for idx,data in enumerate(data_sets):
    data = data.T
    data.sort(0)
    plt.scatter(data[0], data[1], **get_attrs(idx, 'color', 'label',
        'linestyle', **kw))

def PostProcess(opts):
  """Apply desired post-processing to generated plot, and display the result.

  The plot is displayed to the screen, unless :attr:`ofname` is set. In the
  latter case, the plot is saved to disk.

  """
  import matplotlib.pyplot as plt
  if opts.axis:
    plt.axis(map(float, opts.axis.split(',')))
  if opts.title:
    plt.title(opts.title)
  if opts.xlabel:
    plt.axes().set_xlabel(opts.xlabel)
  if opts.ylabel:
    plt.axes().set_ylabel(opts.ylabel)
  if opts.logx:
    plt.gca().set_xscale('log')
  if opts.logy:
    plt.gca().set_yscale('log')
  if opts.command:
    exec opts.command
  if opts.label:
    plt.rcParams['legend.loc'] = 'best'
    plt.legend()
  if opts.ofname:
    plt.savefig(opts.ofname)
  else:
    plt.show()

def PlotFromFile(opts, *fnames):
  """Plot a collection of datasets read from disk.

  :param fnames: Path to datasets.
  :type fnames: list of str
  :param bool post_process: Whether to call :meth:`PostProcess` (default is
     True).

  If needed, :meth:`PreProcess` is called automatically.

  """
  plt = gplot.InitPlot(opts.ofname != None)
  plt.rcParams['lines.linewidth'] = 2
  # Default padding is tuned to figsize of [8,6]
  padding = np.array((.08, .11, .98, .94 if opts.title else .97))
  if opts.padding:
    padding_ = map(float, opts.padding.split(','))[:4]
    padding[:len(padding_)] = padding_
  plt.subplots_adjust(**dict(zip(('left', 'bottom', 'right', 'top'), padding)))
  kw = dict((n,(getattr(opts, n) or '').split(','))
      for n in ('label', 'color', 'linestyle'))
  if opts.plot_type_bar:
    # Bar plots use 1D datasets
    data_sets = map(ReadData1D, fnames)
    data_sets = np.asarray(data_sets)
    # Reshape data_sets from 2D (given by file, line) to 3D (given by file
    # group, file category, line).
    gsize = opts.bar_group_size or 1
    if data_sets.shape[0] % gsize != 0:
      raise ValueError("Number of datasets must be multiple of group size")
    if opts.error_bars:
      data_sets = data_sets.reshape(-1, gsize, data_sets.shape[-1])
    else:
      data_sets = data_sets.reshape(-1, gsize)
    BarPlot(data_sets, **kw)
  elif opts.plot_type_hist:
    # Histograms use 1D datasets
    data_sets = map(ReadData1D, fnames)
    HistPlot(data_sets, **kw)
  else:
    # All other plots use 2D datasets
    data_sets = map(ReadData2D, fnames)
    if opts.plot_type_image:
      ImagePlot(data_sets)
    elif opts.plot_type_scatter:
      ScatterPlot(data_sets, **kw)
    else:
      LinePlot(data_sets, error_bars=opts.error_bars, **kw)
  PostProcess(opts)

def Main(argv = None):
  options = MakeOptions()
  try:
    args = ParseCommandLine(options, argv=argv)
    if options.help.value:
      print >>sys.stderr, "Usage: [options]"
      PrintUsage(options, stream=sys.stderr)
      print (
          "For more option values, see:\n"
          "  http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib"
          ".pyplot.plot\n")
      sys.exit(-1)
    for i in range(len(args)):
      if args[i] == "-":
        args[i] = sys.stdin
    if len(args) < 1:
      args = [ sys.stdin ]
    PlotFromFile(OptValue(options), *args)
  except OptionError, e:
    print >>sys.stderr, "Usage Error (use -h for help): %s." % e
  except KeyboardInterrupt:
    pass

if __name__ == '__main__':
  Main()
