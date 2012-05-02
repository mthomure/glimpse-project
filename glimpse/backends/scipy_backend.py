# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

# Implementation of filter operations using Scipy's ndimage.correlate().

import scipy
from scipy.ndimage import maximum_filter
import numpy as np
from glimpse.util import docstring
from glimpse.backends.backend import IBackend

def Correlate(data, kernel, output = None):
  """Apply a multi-band filter to a set of 2D arrays.

  This is done by applying Scipy's :func:`scipy.ndimage.correlate` to each 2D
  input array and then summing across bands.

  :param data: Input data.
  :type data: ndarray of float
  :param kernel: Kernel array.
  :type kernel: ndarray of float
  :param output: Array in which to store result. If None, a new array will be
     created.
  :type output: ndarray of float
  :return: Output of correlation.
  :rtype: ndarray of float

  """
  assert data.ndim >= 2
  assert data.shape[:-2] == kernel.shape[:-2]
  if output == None:
    output = np.zeros(data.shape[-2:], data.dtype)
  else:
    assert output.shape == data.shape[-2:]
    output[:] = 0
  temp = np.empty_like(output)
  data_ = data.reshape((-1,) + data.shape[-2:])
  kernel_ = kernel.reshape((-1,) + kernel.shape[-2:])
  # TODO consider changing the following to:
  #   scipy.signal.convolve(data_, kernel_, mode='valid')
  for dband, kband in zip(data_, kernel_):
    scipy.ndimage.correlate(dband, kband, output = temp)
    output += temp
  return output

def PruneArray(data, kernel_shape, scaling):
  """Crop correlation results.

  Scipy's :func:`scipy.ndimage.correlate` applies kernel to border units (in the
  last two dimensions), and performs no subsampling. This function returns a
  cropped view of the result array, in which border units have been removed, and
  subsampling has been performed.

  :param data: Result from one or more :func:`scipy.ndimage.correlate` calls
  :type data: ndarray of float
  :param kernel_shape: Shape of the kernel passed to
     :func:`scipy.ndimage.correlate`. If None, then cropping is disabled and
     only subsampling is performed.
  :type kernel_shape: tuple of int
  :param scaling: Subsampling factor.
  :type scaling: positive int
  :return: Cropped, sampled view (not copy) of the input data.
  :rtype: ndarray of float

  """
  if kernel_shape == None:
    # Keep all bands
    band_indices = [slice(None)] * (data.ndim - 2)
    # Perform subsampling of maps
    map_indices = [slice(None, None, scaling)] * 2
  else:
    # Keep all bands
    band_indices = [slice(None)] * (data.ndim - 2)
    # Crop borders of maps by half the kernel width, and perform subsampling.
    assert len(kernel_shape) >= 2
    h, w = kernel_shape[-2:]
    hh = int(h / 2)
    hw = int(w / 2)
    map_indices = [slice(hh, -hh, scaling), slice(hw, -hw, scaling)]
  return data[band_indices + map_indices]

class ScipyBackend(object):
  """:class:`IBackend` implementation using calls to :mod:`scipy` functions."""

  @docstring.copy(IBackend.ContrastEnhance)
  def ContrastEnhance(self, data, kwidth, bias, scaling, out = None):
    assert data.ndim == 2
    kshape = (kwidth, kwidth)
    # x - mu / std
    k = np.ones(kshape)
    k /= k.size
    mu = Correlate(data, k)
    # XXX does it matter whether we add the bias before or after apply sqrt()?
    sigma = np.sqrt(Correlate((data - mu)**2, k) + bias)
    result = (data - mu) / sigma
    return PruneArray(result, kshape, scaling)

  @docstring.copy(IBackend.DotProduct)
  def DotProduct(self, data, kernels, scaling = None, out = None, **ignore):
    assert scaling != None
    assert data.ndim == 3
    assert kernels.ndim == 4
    output_bands = np.empty((kernels.shape[0],) + data.shape[-2:], data.dtype)
    for k, o in zip(kernels, output_bands):
      Correlate(data, k, o)
    output_bands = PruneArray(output_bands, kernels.shape, scaling)
    return output_bands

  @docstring.copy(IBackend.NormDotProduct)
  def NormDotProduct(self, data, kernels, bias = None, scaling = None,
      out = None, **ignore):
    assert bias != None
    assert scaling != None
    assert data.ndim == 3
    assert kernels.ndim == 4
    assert np.allclose(np.array(map(np.linalg.norm, kernels)), 1), \
        "Expected kernels to have unit norm"
    def Op(k, o):
      # dot product of kernel with local input patches
      d = Correlate(data, k, o)
      # norm of local input patches
      n = Correlate(data**2, np.ones(k.shape))
      # use conditional bias
      n[ n < bias ] = bias
      # normalized dot product
      o /= n
    output_bands = np.empty((kernels.shape[0],) + data.shape[-2:], data.dtype)
    for k, o in zip(kernels, output_bands):
      Op(k, o)
    output_bands = PruneArray(output_bands, kernels.shape, scaling)
    if out != None:
      out[:] = output_bands.flat
      return out
    return output_bands

  @docstring.copy(IBackend.Rbf)
  def Rbf(self, data, kernels, beta = None, scaling = None, out = None,
      **ignore):
    assert beta != None
    assert scaling != None
    assert data.ndim == 3
    assert kernels.ndim == 4
    def Op(k, o):
      # (a,a) is squared input patch length
      input_norm = Correlate(data**2, np.ones(k.shape))
      # (b,b) is squared kernel length
      kernel_norm = np.dot(k.flat, k.flat)
      # (a,b) is convolution
      conv = Correlate(data, k)
      # squared distance between input patches and kernel is
      #  ||a-b||^2 = (a,a) + (b,b) - 2(a,b)
      # where a is data and b is kernel.
      square_dist = input_norm + kernel_norm - 2 * conv
      # Gaussian radial basis function
      o[:] = np.exp(-1 * beta * square_dist)
    output_bands = np.empty((kernels.shape[0],) + data.shape[-2:], data.dtype)
    for k, o in zip(kernels, output_bands):
      Op(k, o)
    output_bands = PruneArray(output_bands, kernels.shape, scaling)
    if out != None:
      out[:] = output_bands.flat
      return out
    return output_bands

  @docstring.copy(IBackend.NormRbf)
  def NormRbf(self, data, kernels, bias = None, beta = None, scaling = None,
      out = None, **ignore):
    assert bias != None
    assert beta != None
    assert scaling != None
    nd = self.NormDotProduct(data, kernels, bias, scaling)
    y = np.exp(-2 * beta * (1 - nd))
    if out != None:
      out[:] = y.flat
      return out
    return y

  @docstring.copy(IBackend.LocalMax)
  def LocalMax(self, data, kwidth, scaling, out = None):
    assert len(data.shape) == 3, \
        "Unsupported shape for input data: %s" % (data.shape,)
    kshape = (kwidth, kwidth)
    output = np.empty_like(data)
    for d, o in zip(data, output):
      maximum_filter(d, kshape, output = o)
    output_bands = PruneArray(output, kshape, scaling)
    if out != None:
      out[:] = output_bands.flat
      return out
    return output_bands

  @docstring.copy(IBackend.GlobalMax)
  def GlobalMax(self, data, out = None):
    assert len(data.shape) == 3, \
        "Unsupported shape for input data: %s" % (data.shape,)
    return data.reshape(data.shape[0], -1).max(1, out)

  @docstring.copy(IBackend.OutputMapShapeForInput)
  def OutputMapShapeForInput(self, kheight, kwidth, scaling, iheight, iwidth):
    oheight = iheight / scaling - kheight + 1
    owidth = iwdith / scaling - kheight + 1

  @docstring.copy(IBackend.InputMapShapeForOutput)
  def InputMapShapeForOutput(self, kheight, kwidth, scaling, oheight, owidth):
    iheight = oheight * scaling + kheight - 1
    iwidth = owidth * scaling + kwidth - 1
    return iheight, iwidth

  @docstring.copy(IBackend.PrepareArray)
  def PrepareArray(self, array):
    return array

def ContrastEnhance(data, kwidth, bias, scaling):
  return ScipyBackend().ContrastEnhance(data, kwidth, bias, scaling)

def DotProduct(data, kernels, scaling):
  return ScipyBackend().DotProduct(data, kernels, scaling)

def NormDotProduct(data, kernels, bias, scaling):
  return ScipyBackend().NormDotProduct(data, kernels, bias, scaling)

def Rbf(data, kernels, beta, scaling):
  return ScipyBackend().Rbf(data, kernels, beta, scaling)

def NormRbf(data, kernels, bias, beta, scaling):
  return ScipyBackend().NormRbf(data, kernels, bias, beta, scaling)

def LocalMax(data, kwidth, scaling):
  return ScipyBackend().LocalMax(data, kwidth, scaling)

def GlobalMax(data):
  return ScipyBackend().GlobalMax(data)
