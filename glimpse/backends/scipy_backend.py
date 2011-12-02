# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

# Implementation of filter operations using Scipy's ndimage.correlate().

import scipy
from scipy.ndimage import maximum_filter
import numpy as np

def Correlate(data, kernel, output = None):
  """Apply a multi-band filter to a set of 2-D arrays, by applying Scipy's
  correlate() to each 2-D input array and then summing across bands.
  data -- (N-D) input array
  kernel -- (N-D) kernel array
  output -- (2-D) output array
  RETURN (2-D) array containing output of correlation
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
  """Scipy's correlate() applies kernel to border units (in last two
  dimensions), and performs no subsampling. This function returns a cropped view
  of the result array, in which border units have been removed, and subsampling
  has been performed.
  data -- (N-dim) result array from one or more correlate() calls
  kernel_shape -- (tuple) shape of the kernel passed to correlate(). If None,
                  then cropping is disabled and only subsampling is performed.
  scaling -- (positive int) subsampling factor
  RETURN cropped, sampled view (not copy) of the input data
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

  def ContrastEnhance(self, data, kwidth, bias, scaling, out = None):
    """Apply local contrast stretch to an array.
    data -- (2-D) array of input data
    kwidth -- (int) kernel width
    bias -- (float) additive term in denominator
    scaling -- (positive int) subsampling factor
    out -- (2-D) array in which to store result
    """
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

  def DotProduct(self, data, kernels, scaling, out = None):
    """Convolve an array with a set of kernels.
    data -- (3-D) array of input data
    kernels -- (4-D) array of (3-D) kernels
    scaling -- (positive int) subsampling factor
    out -- (2-D) array in which to store result
    """
    assert data.ndim == 3
    assert kernels.ndim == 4
    output_bands = np.empty((kernels.shape[0],) + data.shape[-2:], data.dtype)
    for k, o in zip(kernels, output_bands):
      Correlate(data, k, o)
    output_bands = PruneArray(output_bands, kernels.shape, scaling)
    return output_bands

  def NormDotProduct(self, data, kernels, bias, scaling, out = None):
    """Convolve an array with a set of kernels, normalizing the response by the
    vector length of the input neighborhood.
    data -- (3-D) array of input data
    kernels -- (4-D) array of (3-D) kernels, where each kernel is expected to
              have unit vector length
    bias -- (float) additive term in denominator
    scaling -- (positive int) subsampling factor
    out -- (2-D) array in which to store result
    """
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

  def Rbf(self, data, kernels, beta, scaling, out = None):
    """Compare kernels to input data using the RBF activation function.
    data -- (3-D) array of input data
    kernels -- (4-D) array of (3-D) kernels
    beta -- (positive float) tuning parameter for radial basis function
    scaling -- (positive int) subsampling factor
    out -- (2-D) array in which to store result
    """
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

  def NormRbf(self, data, kernels, bias, beta, scaling, out = None):
    """Compare kernels to input data using the RBF activation function with
       normed inputs.
    data -- (3-D) array of input data
    kernels -- (4-D) array of (3-D) kernels, where each kernel is expected to
        have unit length
    bias -- (float) additive term in denominator
    beta -- (positive float) tuning parameter for radial basis function
    scaling -- (positive int) subsampling factor
    out -- (2-D) array in which to store result
    """
    nd = self.NormDotProduct(data, kernels, bias, scaling)
    y = np.exp(-2 * beta * (1 - nd))
    if out != None:
      out[:] = y.flat
      return out
    return y

  def LocalMax(self, data, kwidth, scaling, out = None):
    """Convolve maps with local 2-D max filter.
    data -- (3-D) array of input data
    kwidth -- (positive int) kernel width
    scaling -- (positive int) subsampling factor
    out -- (2-D) array in which to store result
    """
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

  def GlobalMax(self, data, out = None):
    """Find the per-band maxima.
    data -- (3-D) array of input data
    out -- (2-D) array in which to store result
    """
    assert len(data.shape) == 3, \
        "Unsupported shape for input data: %s" % (data.shape,)
    return data.reshape(data.shape[0], -1).max(1, out)

  def OutputMapShapeForInput(self, kheight, kwidth, scaling, iheight, iwidth):
    """Given an input map with the given dimensions, compute the shape of the
    corresponding output map.
    kheight -- (positive int) kernel height
    kwidth -- (positive int) kernel width
    scaling -- (positive int) subsampling factor
    iheight -- (positive int) input map height
    iwidth -- (positive int) input map width
    RETURNS (tuple) output map height and width, in that order
    """
    oheight = iheight / scaling - kheight + 1
    owidth = iwdith / scaling - kheight + 1

  def InputMapShapeForOutput(self, kheight, kwidth, scaling, oheight, owidth):
    """The inverse of OutputMapShapeForInput(). Given an output map with the
    given dimensions, compute the shape of the corresponding input map.
    kheight -- (positive int) kernel height
    kwidth -- (positive int) kernel width
    scaling -- (positive int) subsampling factor
    oheight -- (positive int) output map height
    owidth -- (positive int) output map width
    RETURNS (tuple) input map height and width, in that order
    """
    iheight = oheight * scaling + kheight - 1
    iwidth = owidth * scaling + kwidth - 1
    return iheight, iwidth

  def PrepareArray(self, array):
    """Prepare array to be passed to backend methods."""
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
