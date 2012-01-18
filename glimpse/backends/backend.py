# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

# Interface for filter operations.

class InsufficientSizeException(Exception):
  """Exception indicating that the input array was too small (spatially) to
  support the requested operation."""

  def __init__(self, source = None):
    super(Exception, self).__init__()
    self.source = source

class IBackend(object):

  def ContrastEnhance(self, data, kwidth, bias, scaling, out = None):
    """Apply local contrast stretch to an array.
    data -- (2-D) array of input data
    kwidth -- (int) kernel width
    bias -- (float) additive term in denominator
    scaling -- (positive int) subsampling factor
    out -- (2-D) array in which to store result
    """

  def DotProduct(self, data, kernels, scaling, out = None):
    """Convolve an array with a set of kernels.
    data -- (3-D) array of input data
    kernels -- (4-D) array of (3-D) kernels
    scaling -- (positive int) subsampling factor
    out -- (2-D) array in which to store result
    """

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

  def Rbf(self, data, kernels, beta, scaling, out = None):
    """Compare kernels to input data using the RBF activation function.
    data -- (3-D) array of input data
    kernels -- (4-D) array of (3-D) kernels
    beta -- (positive float) tuning parameter for radial basis function
    scaling -- (positive int) subsampling factor
    out -- (2-D) array in which to store result
    """

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

  def LocalMax(self, data, kwidth, scaling, out = None):
    """Convolve maps with local 2-D max filter.
    data -- (3-D) array of input data
    kwidth -- (positive int) kernel width
    scaling -- (positive int) subsampling factor
    out -- (2-D) array in which to store result
    """

  def GlobalMax(self, data, out = None):
    """Find the per-band maxima.
    data -- (3-D) array of input data
    out -- (2-D) array in which to store result
    """

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

  def PrepareArray(self, array):
    """Prepare array to be passed to backend methods."""
