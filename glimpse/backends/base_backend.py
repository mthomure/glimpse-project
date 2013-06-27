# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import numpy as np

from glimpse.backends import _base_backend
from glimpse.backends.misc import BackendError, InputSizeError, IBackend, \
    ACTIVATION_DTYPE
from glimpse.util import docstring

class BaseBackend(object):
  """:class:`IBackend` implementation using vanilla C++ code."""

  @docstring.copy(IBackend.ContrastEnhance)
  def ContrastEnhance(self, data, kwidth, bias, scaling, out = None):
    assert scaling == 1
    result = _base_backend.ContrastEnhance(data, kwidth, kwidth, bias = bias,
        out_data = out)
    if np.isnan(result).any():
      raise BackendError("Found illegal values in result map")
    return result

  @docstring.copy(IBackend.DotProduct)
  def DotProduct(self, data, kernels, scaling = None, out = None, **ignore):
    assert scaling != None
    result = _base_backend.DotProduct(data, kernels, out_data = out,
        scaling = scaling)
    if np.isnan(result).any():
      raise BackendError("Found illegal values in result map")
    return result

  @docstring.copy(IBackend.NormDotProduct)
  def NormDotProduct(self, data, kernels, bias = None, scaling = None,
      out = None, **ignore):
    assert bias != None
    assert scaling != None
    result = _base_backend.NormDotProduct(data, kernels, out_data = out,
        bias = bias, scaling = scaling)
    if np.isnan(result).any():
      raise BackendError("Found illegal values in result map")
    return result

  @docstring.copy(IBackend.Rbf)
  def Rbf(self, data, kernels, beta = None, scaling = None, out = None,
      **ignore):
    assert beta != None
    assert scaling != None
    result = _base_backend.Rbf(data, kernels, out_data = out, beta = beta,
        scaling = scaling)
    if np.isnan(result).any():
      raise BackendError("Found illegal values in result map")
    return result

  @docstring.copy(IBackend.NormRbf)
  def NormRbf(self, data, kernels, bias = None, beta = None, scaling = None,
      out = None, **ignore):
    assert bias != None
    assert beta != None
    assert scaling != None
    result = _base_backend.NormRbf(data, kernels, out_data = out, bias = bias,
        beta = beta, scaling = scaling)
    if np.isnan(result).any():
      raise BackendError("Found illegal values in result map")
    return result

  @docstring.copy(IBackend.LocalMax)
  def LocalMax(self, data, kwidth, scaling, out = None):
    result = _base_backend.LocalMax(data, kheight = kwidth, kwidth = kwidth,
        out_data = out, scaling = scaling)
    if np.isnan(result).any():
      raise BackendError("Found illegal values in result map")
    return result

  @docstring.copy(IBackend.GlobalMax)
  def GlobalMax(self, data, out = None):
    assert len(data.shape) == 3, \
        "Unsupported shape for input data: %s" % (data.shape,)
    if not all(s > 0 for s in data.shape):
      raise InputSizeError
    result = data.reshape(data.shape[0], -1).max(1, out = out)
    if np.isnan(result).any():
      raise BackendError("Found illegal values in result map")
    return result

  @docstring.copy(IBackend.OutputMapShapeForInput)
  def OutputMapShapeForInput(self, kheight, kwidth, scaling, iheight, iwidth):
    oheight, owidth = _base_backend.OutputMapShapeForInput(kheight, kwidth,
        scaling, iheight, iwidth)
    if oheight < 1 or owidth < 1:
      raise InputSizeError
    return oheight, owidth

  @docstring.copy(IBackend.InputMapShapeForOutput)
  def InputMapShapeForOutput(self, kheight, kwidth, scaling, oheight, owidth):
    return _base_backend.InputMapShapeForOutput(kheight, kwidth, scaling,
        oheight, owidth)

  @docstring.copy(IBackend.PrepareArray)
  def PrepareArray(self, array):
    array = array.astype(ACTIVATION_DTYPE)
    # Make sure data is contiguous in memory
    if not array.flags['C_CONTIGUOUS']:
      array = array.copy()
    return array
