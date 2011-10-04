
from scipy.ndimage import correlate, maximum_filter
import numpy as np
from glimpse.core import activation_dtype

def _ScipyOpPruner(kernel_shape, scaling):
  # Scipy's correlate() applies kernel to border pixels. Return a function that
  # cuts this out.
  scaling = int(scaling)
  if len(kernel_shape) == 2:
    h, w = kernel_shape  # height, width
    hh = int(h / 2)
    hw = int(w / 2)
    return lambda x: x[ hh : -hh : scaling, hw : -hw : scaling ]
  elif len(kernel_shape) == 3:
    b, h, w = kernel_shape  # num bands, height, width
    if b % 2 == 0:
      hb = b / 2 - 1
    else:
      hb = int(b / 2)
    hh = int(h / 2)
    hw = int(w / 2)
    return lambda x: x[ hb, hh : -hh : scaling, hw : -hw : scaling ]
  else:
    raise ValueError("Unsupported kernel shape: %s" % (kernel_shape,))

class ScipyBackend(object):

  def ContrastEnhance(self, data, kwidth, bias, scaling):
    """Apply retinal processing to given 2-D data."""
    assert len(data.shape) == 2
    kshape = (kwidth, kwidth)
    prune = _ScipyOpPruner(kshape, scaling = scaling)
    # x - mu / std
    k = np.ones(kshape)
    k /= k.size
    mu = correlate(data, k)
    sigma = np.sqrt(correlate(data**2, k) + bias)
    return prune((data - mu) / sigma)

  def DotProduct(self, data, kernels, scaling):
    """Convolve maps with an array of 2-D kernels."""
    assert len(data.shape) >= 2 and len(data.shape) <= 3
    assert len(kernels.shape) == len(data.shape) + 1
    def Op(k):
      prune = _ScipyOpPruner(k.shape, scaling)
      return prune(correlate(data, k))
    return np.array(map(Op, kernels), activation_dtype)

  def NormDotProduct(self, data, kernels, bias, scaling):
    """Convolve maps with 3-D kernels, normalizing the response by the vector
    length of the input neighborhood.
    data - 3-D array of input data
    kernels - (4-D) array of (3-D) kernels (i.e., all kernels have same shape)
    """
    assert len(data.shape) >= 2 and len(data.shape) <= 3
    assert len(kernels.shape) == len(data.shape) + 1
    def Op(k):
      prune = _ScipyOpPruner(k.shape, scaling)
      # dot product of kernel with local input patches
      d = prune(correlate(data, k))
      # norm of local input patches
      n0 = prune(correlate(data**2, np.ones(k.shape)))
      n = np.sqrt(n0 + bias)
      # normalized dot product
      nd = d / n
      return nd
    return np.array(map(Op, kernels), activation_dtype)

  def Rbf(self, data, kernels, beta, scaling):
    """Compare kernels to input data using the RBF activation function.
    data -- (2- or 3-D) array of input data
    kernels -- (3- or 4-D) array of (2- or 3-D) kernels
    """
    assert len(data.shape) >= 2 and len(data.shape) <= 3
    assert len(kernels.shape) == len(data.shape) + 1
    def Op(k):
      prune = _ScipyOpPruner(k.shape, scaling)
      # ||a-b||^2 = (a,a) + (b,b) - 2(a,b), where a is data and b is kernel
      # (a,a) is squared input patch length
      input_norm = prune(correlate(data**2, np.ones(k.shape)))
      # (b,b) is squared kernel length
      kernel_norm = np.linalg.norm(k)
      # (a,b) is convolution
      conv = prune(correlate(data, k))
      # squared distance between input patches and kernel
      dist = input_norm + kernel_norm - 2 * conv
      # Gaussian radial basis function
      y = np.exp(-2 * beta * dist)
      return y
    return np.array(map(Op, kernels), activation_dtype)

  def NormRbf(self, data, kernels, bias, beta, scaling):
    """Compare kernels to input data using the RBF activation function with
       normed inputs.
    data -- (2- or 3-D) array of input data
    kernels -- (3- or 4-D) array of (2- or 3-D) kernels, where each kernel is
               expected to have unit length
    """
    nd = self.NormDotProduct(data, kernels, bias, scaling)
    y = np.exp(-2 * beta * (1 - nd))
    return y

  def LocalMax(self, data, kwidth, scaling):
    """Convolve maps with local 2-D max filter.
    data - (3-D) array with one 2-D map for each output band
    """
    assert len(data.shape) == 3, \
        "Unsupported shape for input data: %s" % (data.shape,)
    kshape = (kwidth, kwidth)
    prune = _ScipyOpPruner(kshape, scaling)
    def Op(band):
      return prune(maximum_filter(band, kshape))
    return np.array(map(Op, data), activation_dtype)

  def GlobalMax(self, data):
    """Find the per-band maxima.
    data - (3-D) array of one 2-D map for each output band
    """
    assert len(data.shape) == 3, \
        "Unsupported shape for input data: %s" % (data.shape,)
    return data.reshape(data.shape[0], -1).max(1)
    #~ if len(data.shape) == 2:
      #~ return data.max()
    #~ elif len(data.shape) == 3:
      #~ return data.reshape(data.shape[0], -1).max(1)
    #~ raise ValueError("Unsupported shape for input data: %s" % (data.shape,))



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
