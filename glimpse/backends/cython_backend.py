
from glimpse.core import misc

class NotImplemented(Exception):
  pass

class CythonBackend(object):

  def ContrastEnhance(self, data, kwidth, bias):
    """Apply retinal processing to given 2-D data."""
    return misc.BuildRetinalLayer(data, kwidth, bias = bias)

  def DotProduct(self, data, kernels, scaling):
    """Convolve maps with an array of 2-D kernels."""
    raise NotImplemented()

  def NormDotProduct(self, data, kernels, bias, scaling):
    """Convolve maps with 3-D kernels, normalizing the response by the vector
    length of the input neighborhood.
    data - 3-D array of input data
    kernels - (4-D) array of (3-D) kernels (i.e., all kernels have same shape)
    """
    raise NotImplemented()

  def Rbf(self, data, kernels, beta, scaling):
    """Compare kernels to input data using the RBF activation function.
    data -- (2- or 3-D) array of input data
    kernels -- (3- or 4-D) array of (2- or 3-D) kernels
    """
    raise NotImplemented()

  def NormRbf(self, data, kernels, bias, beta, scaling):
    """Compare kernels to input data using the RBF activation function with
       normed inputs.
    data -- (2- or 3-D) array of input data
    kernels -- (3- or 4-D) array of (2- or 3-D) kernels, where each kernel is
               expected to have unit length
    """
    return misc.BuildSimpleLayer(data, kernels, bias = bias, beta = beta,
        scaling = scaling)

  def LocalMax(self, data, kwidth, scaling):
    """Convolve maps with local 2-D max filter.
    data - (3-D) array with one 2-D map for each output band
    """
    odata, _t = misc.BuildComplexLayer(data, kwidth = kwidth, kheight = kwidth,
        scaling = scaling)
    return odata

  def GlobalMax(self, data):
    """Find the per-band maxima.
    data - (3-D) array of one 2-D map for each output band
    """
    assert len(data.shape) == 3, \
        "Unsupported shape for input data: %s" % (data.shape,)
    return data.reshape(data.shape[0], -1).max(1)
