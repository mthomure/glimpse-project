#!/usr/bin/python

# The following methods are useful for visualizing the output of Glimpse using
# matplotlib.

from glimpse import core
import Image
from glimpse import util
from matplotlib.pylab import *
from glimpse.util.gplot import *
import os.path
from scipy.ndimage import filters
import operator

def ScaleData(d):
  d = d.copy()
  d -= d.min()
  d /= d.max()
  d[ d < d.mean() - d.std() ] = 0
  d[ d > d.mean() + d.std() ] = 1

# _sp_y = _sp_x = j = 0
# def MakeSubplot(y, x):
  # global _sp_y, _sp_x, j
  # _sp_y = y
  # _sp_x = x
  # j = 1
# 
# def NextSubplot():
  # global j
  # subplot(_sp_y, _sp_x, j)
  # j += 1

def show_image():
  NextSubplot()
  imshow(i, cmap = cm.gray)
  title('image')
  axis('off')

def show_retina(r, i, options, figure_num = 2):
  mapper = core.transform.CoordinateMapper(options)
  figure(figure_num)
  MakeSubplot(1, 1)
  NextSubplot()
  Show2DArray(r, i, mapper = mapper.MapRetinaToImage, title = 'Retina')

def show_s1(s1, i, options, s1k = None, figure_num = 3):
  mapper = core.transform.CoordinateMapper(options)
  p = 0
  figure(figure_num)
  MakeSubplot(2, 4)
  vmin = s1.min()
  vmax = s1.max()
  for t in range(0, 8):
    if s1k != None:
      k = s1k[t, p]
    else:
      k = None
    NextSubplot()
    Show2DArray(s1[t, p], i, mapper.MapS1ToImage, kernel = k, vmin = vmin, 
        vmax = vmax) 
  suptitle('S1 Layer')

def show_c1(c1, i, options, s1k = None, figure_num = 4):
  mapper = core.transform.CoordinateMapper(options)
  p = 0
  figure(figure_num)
  MakeSubplot(2, 4)
  vmin = c1.min()
  vmax = c1.max()
  for t in range(0, 8):
    if s1k != None:
      k = s1k[t, p]
    else:
      k = None
    NextSubplot()
    Show2DArray(c1[t], i, mapper.MapC1ToImage, kernel = k, vmin = vmin, 
        vmax = vmax)
  suptitle('C1 Layer')

def show_s2(s2, i, options, figure_num = 5):
  mapper = core.transform.CoordinateMapper(options)
  figure(figure_num)
  rows = 1
  cols = min(4, len(s2))
  if len(s2) > cols:
    rows = 2
  MakeSubplot(rows, cols)
  for t in range(0, min(rows*cols, len(s2))):
    NextSubplot()
    Show2DArray(s2[t], i, mapper.MapS2ToImage)
  suptitle('S2 Layer')

def show_s2_kernel(k, figure_num = 6, **args):
  figure(figure_num)
  rows = 1
  cols = min(4, len(k))
  if len(k) > cols:
    rows = 2
  MakeSubplot(rows, cols)
  for t in range(0, min(rows*cols, len(k))):
    NextSubplot()
    imshow(k[t], cmap = cm.gray, **args)
    axis('off')
  suptitle('S2 Kernel Bands')

def show_c2(c2, i, options, figure_num = 7):
  mapper = core.transform.CoordinateMapper(options)
  figure(figure_num)
  rows = 1
  cols = min(4, len(c2))
  if len(c2) > cols:
    rows = 2
  MakeSubplot(rows, cols)
  for t in range(0, min(rows*cols, len(c2))):
    NextSubplot()
    Show2DArray(c2[t], i, mapper.MapC2ToImage)
  suptitle('C2 Layer')

def show_results(r, scale = 0):
  # show_retina(r, i, o)
  show_s1(r.s1_activity[scale], r.image, r.options, s1k = r.s1_kernels)
  show_c1(r.c1_activity[scale], r.image, r.options, s1k = r.s1_kernels)
  # show_s2(r.s2_activity[scale], r.image, r.options)
  # show_c2(r.c2_activity[scale], r.image, r.options)
  draw()

def smoosh(x):
  return x.reshape((-1,) + x.shape[-2:])

def show_s1_reconstructions(results, **args):
  import recon
  
  MakeSubplot(2, 3)
  
  def f(p):
    NextSubplot()
    img = recon.reconstruct_from_s1(results, p)[0]
    imshow(img, cmap = cm.gray)
    axis('off')
    title('L %s' % p)

  f(0)
  f(0.5)
  f(1)
  f(1.5)
  f(2)

if __name__ == '__main__':
  if len(sys.argv) < 3:
    sys.exit("usage: %s IMAGE RESULT-DIR [SCALE]" % sys.argv[0])
  ifname, rdir = sys.argv[1:3]
  scale = 0
  if len(sys.argv) > 3:
    scale = int(sys.argv[3])
  load_data(ifname, rdir, scale = scale)
  show_loaded_data()
  # In case this isn't ipython, leave the plots up until user hits 'enter' key.
  raw_input()

