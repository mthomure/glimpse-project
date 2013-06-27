# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import Image
import numpy as np
import os
from pprint import pformat

__all__ = [
  'ACTIVATION_DTYPE',
  'DEFAULT_BACKEND_NAME',
  'MakeBackend',
  'InputLoadError',
  'InputSource',
  'BackendError',
  'InputSizeError',
  'IBackend',
  ]

#: Element type for an array of Glimpse activation values.
ACTIVATION_DTYPE = np.float32

#: Backend name used by MakeBackend when no name is supplied.
DEFAULT_BACKEND_NAME = "base"

def MakeBackend(name = None):
  """Return an instance of the given backend.

  :param str name: Name of requested backend.

  """
  # Classes are loaded on the fly, in case user has not compiled base_backend.
  if name == None:
    name = os.environ.get('GLIMPSE_BACKEND', DEFAULT_BACKEND_NAME)
  else:
    name = name.lower()
  if name in ("base", "base_backend", "basebackend"):
    from .base_backend import BaseBackend
    return BaseBackend()
  if name in ("scipy", "scipy_backend", "scipybackend"):
    from .scipy_backend import ScipyBackend
    return ScipyBackend()
  raise ValueError("Unknown backend name: %s" % (name,))

class InputLoadError(Exception):
  """Thrown when an input source can not be loaded."""

  def __init__(self, source = None):
    super(InputLoadError, self).__init__()
    #: The *source* of the exception. This is generally an :class:`InputSource`
    #: object.
    self.source = source

  def __str__(self):
    if self.source is not None:
      path = " (%s)" % self.source.image_path
    else:
      path = ""
    return "Failed to process image%s -- file I/O error" % path

  def __repr__(self):
    return "%s(source=%s)" % (type(self).__name__, pformat(self.source))

class InputSource(object):
  """Describes the input to a hierarchical model.

  Example:

  Load a still image from disk.

  >>> source = InputSource(image_path = "/tmp/MyImage.jpg")

  """

  #: Path to the image file.
  image_path = None

  def __init__(self, image_path = None):
    if image_path != None and not isinstance(image_path, basestring):
      raise ValueError("Image path must be a string")
    self.image_path = image_path

  def CreateImage(self):
    """Reads image from this input source.

    :rtype: PIL.Image

    """
    try:
      img = Image.open(self.image_path)
    except IOError:
      raise InputLoadError(source = self)
    return img

  def __str__(self):
    return self.image_path

  def __repr__(self):
    return "%s(image_path=%s)" % (type(self).__name__, pformat(self.image_path))

  def __eq__(self, other):
    return isinstance(other, InputSource) and \
        self.image_path == other.image_path

  def __ne__(self, other):
    return not (self == other)

def _FormatBackendErrorMsg(ex, default_message):
  if ex.source is not None:
    path = " (%s)" % ex.source.image_path
  else:
    path = ""
  suffix = ""
  if ex.layer is not None:
    suffix += " layer %s" % ex.layer
  if ex.scale is not None:
    if suffix:
      suffix = " of%s" % suffix
    suffix = " scale %s%s" % (ex.scale, suffix)
  if suffix:
    suffix = " -- at%s" % suffix
  msg = ex.msg or default_message
  return msg + path + suffix

class BackendError(Exception):
  """Thrown when an unexpected error occurs during backend processing."""

  def __init__(self, msg = None, source = None, layer = None, scale = None):
    super(Exception, self).__init__()
    self.msg = msg
    #: The *source* of the exception. This is generally an :class:`InputSource
    #: <glimpse.models.misc.InputSource>` object.
    self.source = source
    #: The model layer that was being computed.
    self.layer = layer
    #: The scale band that was being computed.
    self.scale = scale

  def __str__(self):
    return _FormatBackendErrorMsg(self, "Failed to process image")

  def __repr__(self):
    return "%s(%s)" % (type(self).__name__, ", ".join("%s=%s" % (k,
        pformat(getattr(self, k))) for k in ('msg', 'source', 'layer',
        'scale')))

class InputSizeError(BackendError):
  """Thrown when an input array was too small (spatially)."""

  def __str__(self):
    return _FormatBackendErrorMsg(self, "Image too small")

class IBackend(object):
  """The interface implemented by all backend implementations."""

  def ContrastEnhance(self, data, kwidth, bias, scaling, out = None):
    """Apply local contrast stretch to an array.

    Given a local input neighborhood :math:`X`, the output is

    .. math::
       y = \\frac{x_c - \mu}{\max(\sigma, \epsilon)}

    where :math:`x_c` is the center of the input neighborhood, :math:`\mu`
    and :math:`\sigma` are the mean and standard deviation of :math:`X`, and
    :math:`\epsilon` is a bias term. This term is used to avoid the
    amplificiation of noise and to ensure a non-zero divisor.

    :param data: Input data.
    :type data: 2D ndarray of float
    :param int kwidth: Kernel width.
    :param float bias: Normalizing term :math:`\epsilon` in denominator.
    :param scaling: Subsampling factor.
    :type scaling: positive int
    :param out: Array in which to store result. If None, a new array will be
       created.
    :type out: 2D ndarray of float
    :rtype: 2D ndarray of float

    """

  def DotProduct(self, data, kernels, scaling = None, out = None, **ignore):
    """Convolve an array with a set of kernels.

    This function compares each kernel :math:`W` and input neigborhood :math:`X`
    using a dot product, where the output is given by

    .. math::
       y = X^T W \, ,

    where :math:`X^T` denotes the matrix transpose.

    :param data: Input data.
    :type data: 3D ndarray of float
    :param kernels: Array of 3D kernels.
    :type kernels: 4D ndarray of float
    :param scaling: Subsampling factor.
    :type scaling: positive int
    :param out: Array in which to store result. If None, a new array will be
       created.
    :type out: 3D ndarray of float
    :rtype: 3D ndarray of float

    """

  def NormDotProduct(self, data, kernels, bias = None, scaling = None,
      out = None, **ignore):
    """Convolve an array with a set of kernels, normalizing the response by the
    vector length of the input neighborhood.

    The output for each kernel :math:`W` and input neighborhood :math:`X` is
    given by

    .. math::
       y = \\frac{X^T W}{\\left\Vert X \\right\Vert \\left\Vert W \\right\Vert} \, ,

    where :math:`\\left\Vert \cdot \\right\Vert` denotes the Euclidean norm.

    :param data: Input data.
    :type data: 3D ndarray of float
    :param kernels: Array of 3D kernels, where each kernel is expected to have
       unit vector length.
    :type kernels: 4D ndarray of float
    :param float bias: Threshold for denominator.
    :param scaling: Subsampling factor.
    :type scaling: positive int
    :param out: Array in which to store result. If None, a new array will be
       created.
    :type out: 3D ndarray of float
    :rtype: 3D ndarray of float

    """

  def Rbf(self, data, kernels, beta = None, scaling = None, out = None,
      **ignore):
    """Compare kernels to input data using the RBF activation function.

    The output for each kernel :math:`W` and input neighborhood :math:`X` is
    given by

    .. math::
       y = \exp \\left\{ - \\beta \\left\Vert X - W \\right\Vert ^2 \\right\} \, ,

    where :math:`\\beta` controls the sensitivity of the RBF.

    :param data: Input data.
    :type data: 3D ndarray of float
    :param kernels: Array of 3D kernels.
    :type kernels: 4D ndarray of float
    :param float beta: Tuning parameter for radial basis function.
    :param scaling: Subsampling factor.
    :type scaling: positive int
    :param out: Array in which to store result. If None, a new array will be
       created.
    :type out: 3D ndarray of float
    :rtype: 3D ndarray of float

    """

  def NormRbf(self, data, kernels, bias = None, beta = None, scaling = None,
      out = None, **ignore):
    """Compare kernels to input data using the RBF activation function with
       normed inputs.

    The output for each kernel :math:`W` and input neighborhood :math:`X` is
    given by

    .. math::
       y = \exp \\left\{ - 2\\beta \\left(1 - \\text{NDP}(X, W) \\right) \\right\} \, ,

    where :math:`\\text{NDP}(\cdot)` is the :meth:`normalized dot product
    <NormDotProduct>`.

    :param data: Input data.
    :type data: 3D ndarray of float
    :param kernels: Array of 3D kernels, where each kernel is expected to
       have unit length.
    :type kernels: 4D ndarray of float
    :param float bias: Additive term in denominator.
    :param float beta: Tuning parameter for radial basis function.
    :param scaling: Subsampling factor.
    :type scaling: positive int
    :param out: Array in which to store result. If None, a new array will be
       created.
    :type out: 3D ndarray of float
    :rtype: 3D ndarray of float

    """

  def LocalMax(self, data, kwidth, scaling, out = None):
    """Convolve maps with local 2-D max filter.

    The output for each local input neighborhood :math:`X` is

    .. math::
       y = max_i \ x_i \, .

    :param data: Input data.
    :type data: 3D ndarray of float
    :param kwidth: Width of pooling neighborhood.
    :type kwidth: positive int
    :param scaling: Subsampling factor.
    :type scaling: positive int
    :param out: Array in which to store result. If None, a new array will be
       created.
    :type out: 3D ndarray of float
    :rtype: 3D ndarray of float

    """

  def GlobalMax(self, data, out = None):
    """Find the per-band maxima.

    :param data: Input data.
    :type data: 3D ndarray of float
    :param out: Array in which to store result. If None, a new array will be
       created.
    :type out: 1D ndarray of float
    :rtype: 1D ndarray of float

    """

  def OutputMapShapeForInput(self, kheight, kwidth, scaling, iheight, iwidth):
    """Given an input map with the given dimensions, compute the shape of the
    corresponding output map.

    :param kheight: Kernel height.
    :type kheight: positive int
    :param kwidth: Kernel width.
    :type kwidth: positive int
    :param scaling: Subsampling factor.
    :type scaling: positive int
    :param iheight: Input map height.
    :type iheight: positive int
    :param iwidth: Input map width.
    :type iwidth: positive int
    :return: Output map height and width, in that order.
    :rtype: 2-tuple of int

    """

  def InputMapShapeForOutput(self, kheight, kwidth, scaling, oheight, owidth):
    """Given an output map with the given dimensions, compute the shape of the
    corresponding input map.

    This is the inverse of :meth:`OutputMapShapeForInput`.

    :param kheight: Kernel height.
    :type kheight: positive int
    :param kwidth: Kernel width.
    :type kwidth: positive int
    :param scaling: Subsampling factor.
    :type scaling: positive int
    :param oheight: Output map height.
    :type oheight: positive int
    :param owidth: Output map width.
    :type owidth: positive int
    :return: Input map height and width, in that order.
    :rtype: 2-tuple of int

    """
  def PrepareArray(self, array):
    """Prepare array to be passed to backend methods.

    :param array: Array to be prepared, which will *not* be modified.
    :type array: ndarray of float
    :rtype: ndarray of float

    """
