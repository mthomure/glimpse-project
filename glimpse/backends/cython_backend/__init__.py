# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

# Implementation of filter operations using custom C++ code.

import filter
from glimpse.util import ACTIVATION_DTYPE

class CythonBackend(object):

  def ContrastEnhance(self, data, kwidth, bias, scaling):
    """Apply local contrast stretch to an array.
    data -- (2-D) array of input data
    kwidth -- (int) kernel width
    bias -- (float) additive term in denominator
    scaling -- (positive int) subsampling factor
    """
    assert scaling == 1
    return filter.ContrastEnhance(data, kwidth, kwidth, bias = bias)

  def DotProduct(self, data, kernels, scaling):
    """Convolve an array with a set of kernels.
    data -- (3-D) array of input data
    kernels -- (4-D) array of (3-D) kernels
    scaling -- (positive int) subsampling factor
    """
    return filter.DotProduct(data, kernels, scaling = scaling)

  def NormDotProduct(self, data, kernels, bias, scaling):
    """Convolve an array with a set of kernels, normalizing the response by the
    vector length of the input neighborhood.
    data -- (3-D) array of input data
    kernels -- (4-D) array of (3-D) kernels, where each kernel is expected to
              have unit vector length
    bias -- (float) additive term in denominator
    scaling -- (positive int) subsampling factor
    """
    return filter.NormDotProduct(data, kernels, bias = bias, scaling = scaling)

  def Rbf(self, data, kernels, beta, scaling):
    """Compare kernels to input data using the RBF activation function.
    data -- (3-D) array of input data
    kernels -- (4-D) array of (3-D) kernels
    beta -- (positive float) tuning parameter for radial basis function
    scaling -- (positive int) subsampling factor
    """
    return filter.Rbf(data, kernels, beta = beta, scaling = scaling)

  def NormRbf(self, data, kernels, bias, beta, scaling):
    """Compare kernels to input data using the RBF activation function with
       normed inputs.
    data -- (3-D) array of input data
    kernels -- (4-D) array of (3-D) kernels, where each kernel is expected to
        have unit length
    bias -- (float) additive term in denominator
    beta -- (positive float) tuning parameter for radial basis function
    scaling -- (positive int) subsampling factor
    """
    return filter.NormRbf(data, kernels, bias = bias, beta = beta,
        scaling = scaling)

  def LocalMax(self, data, kwidth, scaling):
    """Convolve maps with local 2-D max filter.
    data -- (3-D) array of input data
    kwidth -- (positive int) kernel width
    scaling -- (positive int) subsampling factor
    """
    return filter.LocalMax(data, kwidth = kwidth, kheight = kwidth,
        scaling = scaling)

  def GlobalMax(self, data):
    """Find the per-band maxima.
    data -- (3-D) array of input data
    """
    assert len(data.shape) == 3, \
        "Unsupported shape for input data: %s" % (data.shape,)
    return data.reshape(data.shape[0], -1).max(1)

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
    return filter.OutputMapShapeForInput(kheight, kwidth, scaling, iheight,
        iwidth)

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
    return filter.InputMapShapeForOutput(kheight, kwidth, scaling, oheight,
        owidth)

  def PrepareArray(self, array):
    """Prepare array to be passed to backend methods."""
    array = array.astype(ACTIVATION_DTYPE)
    # Make sure data is contiguous in memory
    if not array.flags['C_CONTIGUOUS']:
      array = array.copy()
    return array

def ContrastEnhance(data, kwidth, bias, scaling):
  return CythonBackend().ContrastEnhance(data, kwidth, bias, scaling)

def DotProduct(data, kernels, scaling):
  return CythonBackend().DotProduct(data, kernels, scaling)

def NormDotProduct(data, kernels, bias, scaling):
  return CythonBackend().NormDotProduct(data, kernels, bias, scaling)

def Rbf(data, kernels, beta, scaling):
  return CythonBackend().Rbf(data, kernels, beta, scaling)

def NormRbf(data, kernels, bias, beta, scaling):
  return CythonBackend().NormRbf(data, kernels, bias, beta, scaling)

def LocalMax(data, kwidth, scaling):
  return CythonBackend().LocalMax(data, kwidth, scaling)

def GlobalMax(data):
  return CythonBackend().GlobalMax(data)
