# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import filter
from glimpse.backends.backend import InsufficientSizeException, IBackend
from glimpse.util import ACTIVATION_DTYPE
from glimpse.util import docstring

class CythonBackend(object):
  """:class:`IBackend` implementation using custom C++ code."""

  @docstring.copy(IBackend.ContrastEnhance)
  def ContrastEnhance(self, data, kwidth, bias, scaling, out = None):
    assert scaling == 1
    return filter.ContrastEnhance(data, kwidth, kwidth, bias = bias,
        out_data = out)

  @docstring.copy(IBackend.DotProduct)
  def DotProduct(self, data, kernels, scaling = None, out = None, **ignore):
    assert scaling != None
    return filter.DotProduct(data, kernels, out_data = out, scaling = scaling)

  @docstring.copy(IBackend.NormDotProduct)
  def NormDotProduct(self, data, kernels, bias = None, scaling = None,
      out = None, **ignore):
    assert bias != None
    assert scaling != None
    return filter.NormDotProduct(data, kernels, out_data = out, bias = bias,
        scaling = scaling)

  @docstring.copy(IBackend.Rbf)
  def Rbf(self, data, kernels, beta = None, scaling = None, out = None,
      **ignore):
    assert beta != None
    assert scaling != None
    return filter.Rbf(data, kernels, out_data = out, beta = beta,
        scaling = scaling)

  @docstring.copy(IBackend.NormRbf)
  def NormRbf(self, data, kernels, bias = None, beta = None, scaling = None,
      out = None, **ignore):
    assert bias != None
    assert beta != None
    assert scaling != None
    return filter.NormRbf(data, kernels, out_data = out, bias = bias,
        beta = beta, scaling = scaling)

  @docstring.copy(IBackend.LocalMax)
  def LocalMax(self, data, kwidth, scaling, out = None):
    return filter.LocalMax(data, kheight = kwidth, kwidth = kwidth,
        out_data = out, scaling = scaling)

  @docstring.copy(IBackend.GlobalMax)
  def GlobalMax(self, data, out = None):
    assert len(data.shape) == 3, \
        "Unsupported shape for input data: %s" % (data.shape,)
    if not all(s > 0 for s in data.shape):
      raise InsufficientSizeException
    return data.reshape(data.shape[0], -1).max(1, out = out)

  @docstring.copy(IBackend.OutputMapShapeForInput)
  def OutputMapShapeForInput(self, kheight, kwidth, scaling, iheight, iwidth):
    oheight, owidth = filter.OutputMapShapeForInput(kheight, kwidth, scaling,
        iheight, iwidth)
    if oheight < 1 or owidth < 1:
      raise InsufficientSizeException
    return oheight, owidth

  @docstring.copy(IBackend.InputMapShapeForOutput)
  def InputMapShapeForOutput(self, kheight, kwidth, scaling, oheight, owidth):
    return filter.InputMapShapeForOutput(kheight, kwidth, scaling, oheight,
        owidth)

  @docstring.copy(IBackend.PrepareArray)
  def PrepareArray(self, array):
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
