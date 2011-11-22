# Code to implement Object-Distractor Difference (ODD) Kernels under Glimpse.

from glimpse import core
from glimpse.core import transform
from glimpse import util
from math import exp
import numpy as np
import odd_core as _core
import os

# def remap_to_dotprod(x, beta, out = None):
  # """Map a set of activations computed via normalized RBF back to the 
  # corresponding dot product."""
  # # out = 1 + log(x) / (2*beta)
  # out = np.log(x, out)
  # out /= 2 * beta
  # out += 1
  # return out

# maybe use tau = 0.05 ?
def ThresholdActivity(x, tau, beta, out = None):
  """Shifts zero point to exp(-2 * beta) and applies threshold.
  x -- N-D array of activity maps
  tau -- threshold on shifted activity (e.g., 0.03)"""
  if out == None:
    out = x.copy()
  else:
    out.reshape((-1,))[:] = x.reshape(x.size)
  out -= exp(-2 * beta)
  out[ out < tau ] = 0
  return out

def LearnWeights(kwidth, activities, step = 25, tau = 0.03, beta = 1.0, 
    use_max = False):
  """Compute co-occurrence statistics for patches from the given activity 
  maps. Weights are learned for each orientation independently, and only for
  patches in which the center unit at the corresponding orientation is "on" -- 
  as determined by thresholding the activty maps.
  kwidth -- length of weight matrix (on a side)
  activities -- list (or array) of 3D activity maps giving C1 activity, or S1 
                activity that has been pooled over phase, for each image.
  step -- subsampling factor for choosing patches from activity arrays (e.g., a
          step of 1 takes patches from every location, while a step of 2 takes 
          patches from every other location.
  tau -- threshold for determining when a unit is "on"
  beta -- the S1 beta value used when computing activity from the image
  use_max -- compute co-occurrence information only for the maximal orientation
             at a given location. otherwise, co-occurrence information is 
             collected for all orientations that are "on".
  """
  if len(activities) == 0:
    return
  nbands = activities[0].shape[0]
  weights = np.zeros((nbands, nbands, kwidth, kwidth), np.float32)
  for a in activities:
    assert len(a.shape) == 3, "Activity map has wrong shape: expected " \
        "3 dimensions, got %d" % len(a.shape)
    ap = ThresholdActivity(a, tau, beta)
    if use_max:
      ap = np.rollaxis(ap, 0, 3)
      _core.UpdateWeightsWithMax(weights, ap, step)
    else:
      _core.UpdateWeights(weights, ap, step)
  return weights

def LearnCov(kwidth, activities, means, step = 25):
  """Compute co-occurrence statistics for patches from the given activity 
  maps. Weights are learned for each orientation independently, and only for
  patches in which the center unit at the corresponding orientation is "on" -- 
  as determined by thresholding the activty maps.
  kwidth -- length of weight matrix (on a side)
  activities -- list (or array) of 3D activity maps giving C1 activity, or S1 
                activity that has been pooled over phase, for each image.
  step -- subsampling factor for choosing patches from activity arrays (e.g., a
          step of 1 takes patches from every location, while a step of 2 takes 
          patches from every other location.
  tau -- threshold for determining when a unit is "on"
  beta -- the S1 beta value used when computing activity from the image
  """
  if len(activities) == 0:
    return
  nbands = activities[0].shape[0]
  cov = np.zeros((nbands, nbands, kwidth, kwidth), np.float32)
  nelements = 0  # number of total patches analyzed
  for a in activities:
    assert len(a.shape) == 3, "Activity map has wrong shape: expected " \
        "3 dimensions, got %d" % len(a.shape)
    _core.UpdateCovariance(cov, a, means, step)
    height, width = a.shape[1:3]
    nelements += (height - kwidth + 1) * (width - kwidth + 1)
  # Covariance is expectation, so divide by total number of S1 patches.
  cov /= nelements
#  for i in range(nbands):
#    for j in range(nbands):
#      cov[i, j] /= stds[i] * stds[j]
  return cov

def LearnThresholdedCov(kwidth, activities, means, step = 25, beta = 1.0, 
    tau = 0.001):
  """Compute co-occurrence statistics for patches from the given activity 
  maps. Weights are learned for each orientation independently, and only for
  patches in which the center unit at the corresponding orientation is "on" -- 
  as determined by thresholding the activty maps.
  kwidth -- length of weight matrix (on a side)
  activities -- list (or array) of 3D activity maps giving C1 activity, or S1 
                activity that has been pooled over phase, for each image.
  step -- subsampling factor for choosing patches from activity arrays (e.g., a
          step of 1 takes patches from every location, while a step of 2 takes 
          patches from every other location.
  tau -- threshold for determining when a unit is "on"
  beta -- the S1 beta value used when computing activity from the image
  """
  if len(activities) == 0:
    return
  zero_point = exp(-2 * beta)
  nbands = activities[0].shape[0]
  cov = np.zeros((nbands, nbands, kwidth, kwidth), np.float32)
  counts = np.zeros(nbands, np.int32)
  for a in activities:
    assert len(a.shape) == 3, "Activity map has wrong shape: expected " \
        "3 dimensions, got %d" % len(a.shape)
    _core.UpdateThresholdedCovariance(cov, a, means, step, 
        zero_point = zero_point, tau = tau, counts = counts)
  # Covariance is expectation, so divide by total number of S1 patches.
  for cov_band, counts_band in zip(cov, counts):
    cov_band /= counts_band
  return cov

def OrientationField(weights, options, pad = 0, width = None):
  """Visualize the weight matrix for a lateral kernel by drawing a line at the 
  maximal orientation for each location. The intensity of the line gives the 
  relative value of the corresponding weight.
  weights -- weight matrix (with multiple bands) conditioned on a single 
             orientation.
  options -- parameters used for the feedforward transformations
  pad -- number of pixels to leave between each drawn line
  RETURNS: 2D matrix to be plotted, e.g., with imshow() 
  """
  if width == None:
    width = 3
  num_orientations, h, w = weights.shape
  assert num_orientations == options['s1_num_orientations']
  kw = options['s1_kwidth']
  hw = kw / 2
  pad += kw
  lines = [ transform.DrawGaborAsLine(i, options, width = width) for i in 
      range(num_orientations) ]
  out = np.zeros((pad * h, pad * w), np.float32)

  # suppress the set of non-maximal orientations at each location
  weights_p = weights.copy()
  weights_p[ weights < weights.max(0) ] = 0
  weights = weights_p

  def draw1():
    # Draws weights as overlay of different lines. 
    for y, x in np.ndindex(weights.shape[-2:]):
      ws = weights[:, y, x]
      y *= pad
      x *= pad
      # Print lines in reverse order of weight
      for b in ws.argsort()[::-1]:
        t = out[ y:y+kw, x:x+kw ]
        l = lines[b]
        # Draw line over existing subfield
        t[ l > 0 ] = l[ l > 0 ] * ws[b]

  def draw2():
    # Draw weights as super-position of lines. This probably doesn't make sense
    # unless non-maximal edge orientations have been suppressed above.
    for idx, w in np.ndenumerate(weights):
      b, y, x = idx
      y *= pad
      x *= pad
      out[ y:y+kw, x:x+kw ] += lines[b] * w

  draw2()
  return out

def DrawOddKernels(pos_class_weights, pos_class_name, neg_class_weights, 
    neg_class_name, options, pad = 0, width = 4, odir = None, **args):
  from matplotlib import pyplot, cm
  s1k = core.MakeS1Kernels(options)[:, 0]
  pyplot.figure(1)
  DrawOrientationFields(pos_class_weights, options, annotations = s1k, 
      cmap = cm.spectral, **args)
  pyplot.suptitle("Lateral Kernels -- %s" % pos_class_name)
  if odir != None:
    pyplot.savefig(os.path.join(odir, "odd-kernels-%s.png" % pos_class_name))
  pyplot.figure(2)
  DrawOrientationFields(neg_class_weights, options, annotations = s1k, 
      cmap = cm.spectral, **args)
  pyplot.suptitle("Lateral Kernels -- %s" % neg_class_name)
  if odir != None:
    pyplot.savefig(os.path.join(odir, "odd-kernels-%s.png" % neg_class_name))
  odd_weights = pos_class_weights - neg_class_weights
  pyplot.figure(3)
  DrawOrientationFields(odd_weights, options, annotations = s1k, 
      cmap = cm.seismic, **args)
  pyplot.suptitle("Lateral Kernels -- %s/%s Difference" % (pos_class_name, 
      neg_class_name))
  if odir != None:
    pyplot.savefig(os.path.join(odir, "odd-kernels-difference.png"))

def DrawOrientationFields(weights, options, pad = 0, annotations = None, 
    width = None, **args):
  from glimpse.util import gplot
  fields = np.array([ OrientationField(w, options, pad = pad, width = width) 
      for w in weights ])
  if fields.min() < 0:
    vmax = max(abs(fields.min()), fields.max()) 
    vmin = -vmax
  else:
    vmax = fields.max()
    vmin = 0
  args = util.MergeDict(args, vmin = vmin, vmax = vmax, cols = 5)
  if annotations == None:
    annotations = [ None ] * len(fields)
  plot.Show2DArrayList(fields, annotations = annotations, **args)
  return fields

def ApplyOddKernels(weights, idata, options, scale = 0.05, tau = 0.05, step = 1, 
      iterations = 4):
  """
  scale -- controls amount of change from one iteration to the next
  """
  idata = idata - exp(-2 * options['s1_beta'])
  weights = weights / max(abs(weights.min()), weights.max())
  weights *= scale
  weights += 1
  yield idata.copy()
  for i in range(iterations):
    _core.ApplyWeights(weights, idata, tau, step)
    yield idata.copy()
  raise StopIteration


# PatchIterator : Image -> [Patch]
def PatchIterator(
    data,  # 2-D array of input data
    pwidth,  # size of square patch on a side
    step = 1,  # subsampling factor
    ):
  height, width = data.shape
  for i in range(0, height - pwidth + 1, step):
    for j in range(0, width - pwidth + 1, step):
      yield data[i : i + pwidth, j : j + pwidth].reshape(-1)
  raise StopIteration

def ToDotProduct(s1, options = None):
  if options == None:
    options = core.MakeDefaultOptions()
  beta = options['s1_beta']
  return 1 + 1 / (2.0 * beta) * np.log(s1)

def LearnCovInMemory(s1s, normalize = False, pool_phase = True, 
    as_dotprod = False, options = None, kwidth = 21):
  """Compute the pairwise covariance of the center unit with other units in each
  patch.
  s1s -- list of (4-D) S1 activity arrays
  normalize -- scale the covariance by the standard deviations (i.e., compute
               correlation coefficients)
  """
  assert len(s1s) > 0
  if options == None:
    options = core.MakeDefaultOptions()
  if pool_phase:
    # pool over phase
    s1s = [ s1.max(1) for s1 in s1s ]
  else:
    # combine bands for all phases
    norientations, nphases = s1s[0].shape[:2]
    s1s = [ s1.reshape((-1,) + s1.shape[-2:]) for s1 in s1s ]
  # convert RBF to dot product
  if as_dotprod:
    s1s = [ ToDotProduct(s1, options) for s1 in s1s ]
  # get patches for each band of each image
  patches = np.array([ [list(PatchIterator(b, kwidth)) for b in s1] 
      for s1 in s1s ])
  # rearrange patches to be ordered by image+location, then by band
  patches = np.rollaxis(patches, 2, 1)
  # concatenate corresponding patches across band, and squeeze image+location to
  # a single axis
  nimages, nlocs, nbands, psize = patches.shape
  patches = patches.reshape(nimages * nlocs, nbands * psize)
  # compute covariance matrix, with variables separated by columns
  if normalize:
    c = np.corrcoef(patches, rowvar = 0)
  else:
    c = np.cov(patches, rowvar = 0)
  c = c.reshape(nbands, kwidth, kwidth, nbands, kwidth, kwidth)
  hw = kwidth / 2
  # get the covariance of center unit wrt each other unit
  c = c[:, hw, hw]
  if not pool_phase:
    new_shape = (norientations, nphases, norientations, nphases, kwidth, kwidth)
    c = c.reshape(new_shape)
  return c

