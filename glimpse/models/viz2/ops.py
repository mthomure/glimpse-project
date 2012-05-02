# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import copy
from glimpse import backends
from glimpse.models.misc import ImageLayerFromInputArray, SampleC1Patches
from glimpse.util import kernel
from glimpse.util import ACTIVATION_DTYPE
import numpy as np
from params import Params

class ModelOps(object):
  """Base class for a Glimpse model based on PANN.

  This class implements all single-layer transformations.

  """

  #: The parameters type associated with this model.
  ParamsClass = Params

  def __init__(self, backend = None, params = None):
    """Create new object.

    :param IBackend backend: Implementation of backend operations, such as
       dot-products.
    :param Params params: Model configuration, such as S-unit kernel
       widths.

    """
    if backend == None:
      backend = backends.MakeBackend()
    else:
      backend = copy.copy(backend)
    if params == None:
      params = self.ParamsClass()
    else:
      if not isinstance(params, self.ParamsClass):
        raise ValueError("Params object has wrong type: expected %s, got %s" % \
            (self.ParamsClass, type(params)))
      params = copy.copy(params)
    self.backend = backend
    self.params = params
    self._s1_kernels = None
    self._s2_kernels = None

  @property
  def s1_kernel_shape(self):
    """The expected shape of the S1 kernels array, including band structure.

    :rtype: tuple of int

    """
    p = self.params
    return p.num_scales, p.s1_num_orientations, p.s1_num_phases, p.s1_kwidth, \
        p.s1_kwidth

  @property
  def s1_kernels(self):
    """The S1 kernels array, which are generated if not set."""
    # if kernels array is empty, then generate it using current model parameters
    if self._s1_kernels == None:
      p = self.params
      self._s1_kernels = kernel.MakeMultiScaleGaborKernels(
        kwidth = p.s1_kwidth, num_scales = p.num_scales,
        num_orientations = p.s1_num_orientations, num_phases = p.s1_num_phases,
        shift_orientations = True, scale_norm = True)
    return self._s1_kernels

  @s1_kernels.setter
  def s1_kernels(self, kernels):
    if kernels == None:
      self._s1_kernels = None
      return
    if kernels.shape != self.s1_kernel_shape:
      raise ValueError("S1 kernels have wrong shape: expected %s, got %s" % \
          (self.s1_kernel_shape, kernels.shape))
    if self.params.s1_operation in ('NormDotProduct', 'NormRbf'):
      if not np.allclose(np.array(map(np.linalg.norm, kernels)), 1):
        raise ValueError("S1 kernels are not normalized")
    if np.isnan(kernels).any():
      raise ValueError("S1 kernels contain NaN")
    self._s1_kernels = kernels.astype(ACTIVATION_DTYPE)

  @property
  def s2_kernel_shape(self):
    """The expected shape of a single S2 kernel, *without* band structure.

    :rtype: tuple of int

    """
    p = self.params
    return p.s1_num_orientations, p.s2_kwidth, p.s2_kwidth

  @property
  def s2_kernels(self):
    """The S2 kernels array."""
    return self._s2_kernels

  @s2_kernels.setter
  def s2_kernels(self, kernels):
    if kernels == None:
      self._s2_kernels = None
      return
    # Check kernel shape
    if kernels.shape[1:] != self.s2_kernel_shape:
      raise ValueError("S2 kernels have wrong shape: expected "
          " (*, %s, %s, %s), got %s" % (self.s2_kernel_shape + \
          [kernels.shape]))
    # Check kernel values
    if self.params.s2_operation in ('NormDotProduct', 'NormRbf'):
      if not np.allclose(np.array(map(np.linalg.norm, kernels)), 1):
        raise ValueError("S2 kernels are not normalized")
    if np.isnan(kernels).any():
      raise ValueError("S2 kernels contain NaN")
    self._s2_kernels = kernels.astype(ACTIVATION_DTYPE)

  def BuildImageFromInput(self, input_):
    """Create the initial image layer from some input.

    :param input_: Input data. If array, values should lie in the range [0, 1].
    :type input_: PIL.Image or 2D ndarray
    :returns: image layer data
    :rtype: 2D ndarray of float

    """
    return ImageLayerFromInputArray(input_, self.backend)

  def BuildRetinaFromImage(self, img):
    """Compute retinal layer activity from the input image.

    :param img: Image data
    :type: 2D ndarray of float
    :rtype: 2D ndarray of float

    """
    p = self.params
    if not p.retina_enabled:
      return img
    retina = self.backend.ContrastEnhance(img, kwidth = p.retina_kwidth,
        bias = p.retina_bias, scaling = 1)
    return retina

  def BuildS1FromRetina(self, retina):
    """Apply S1 processing to some existing retinal layer data.

    .. note::

       This method pools over phase, so the output array has only scale and
       orientation bands.

    :param retina: Result of retinal layer processing.
    :type retina: 2D ndarray of float
    :rtype: 4D ndarray of float

    """
    # Reshape retina to be 3D array
    p = self.params
    retina_ = retina.reshape((1,) + retina.shape)
    # Reshape kernel array to be 4-D: scale, index, 1, y, x
    s1_kernels = self.s1_kernels.reshape((-1, 1, p.s1_kwidth, p.s1_kwidth))
    backend_op = getattr(self.backend, p.s1_operation)
    s1_ = backend_op(retina_, s1_kernels, bias = p.s1_bias, beta = p.s1_beta,
        scaling = p.s1_scaling)
    # Reshape S1 to be 5D array
    s1 = s1_.reshape((p.num_scales, p.s1_num_orientations, p.s1_num_phases) + \
        s1_.shape[-2:])
    # Pool over phase.
    s1 = s1.max(2)
    return s1

  def BuildC1FromS1(self, s1s):
    """Compute the C1 layer activity from multi-scale S1 activity.

    :param s1s: S1 activity for each scale.
    :type s1s: 4D ndarray of float
    :returns: C1 activity, with one array per scale.
    :rtype: list of 3D ndarray of float

    """
    p = self.params
    c1s = [ self.backend.LocalMax(s1, kwidth = p.c1_kwidth,
        scaling = p.c1_scaling) for s1 in s1s ]
    if p.c1_whiten:
      #~ # DEBUG: use this to whiten over orientation only
      #~ map(Whiten, c1s)
      # DEBUG: use this to whiten over scale AND orientation concurrently. PANN
      # and the old Viz2 model used this.
      c1s = np.array(c1s, ACTIVATION_DTYPE)
      c1_shape = c1s.shape
      c1s = c1s.reshape((-1,) + c1s.shape[-2:])
      Whiten(c1s)
      c1s = c1s.reshape(c1_shape)
    return c1s

  def BuildS2FromC1(self, c1s):
    """Compute the S2 layer activity from multi-scale C1 activity.

    :param c1s: C1 activity
    :type c1s: 4D ndarray of float, or list of 3D ndarray of float
    :returns: S2 activity for each scale
    :rtype: 4D ndarray of float

    """
    if self.s2_kernels == None:
      raise Exception("Need S2 kernels to compute S2 layer activity, but none "
          "were specified.")
    if len(c1s) == 0:
      return []
    p = self.params
    c1s = np.array(c1s, copy = False)
    # Get the spatial extent of a map. Note that this assumes that every C1 map
    # has same spatial extent.
    height, width = c1s[0].shape[-2:]
    # Get the shape of each S2 map
    s2_shape = self.backend.OutputMapShapeForInput(p.s2_kwidth, p.s2_kwidth,
        p.s2_scaling, height, width)
    # Get the shape of the full S2 activity array
    s2_shape = (p.num_scales, len(self.s2_kernels)) + s2_shape
    s2s = np.empty(s2_shape, ACTIVATION_DTYPE)
    backend_op = getattr(self.backend, p.s2_operation)
    for scale in range(p.num_scales):
      c1 = c1s[scale]
      s2 = s2s[scale]
      backend_op(c1, self.s2_kernels, bias = p.s2_bias,
          beta = p.s2_beta, scaling = p.s2_scaling, out = s2)
    return s2s

  def BuildC2FromS2(self, s2s):
    """Compute the C2 layer activity from multi-scale S2 activity.

    :param s2s: S2 activity
    :type: s2s: 4D array
    :returns: C2 activity for each scale and prototype
    :rtype: 2D ndarray of float

    """
    c2s = map(self.backend.GlobalMax, s2s)
    c2s = np.array(c2s, ACTIVATION_DTYPE)
    return c2s

  def BuildItFromC2(self, c2s):
    """Compute the IT layer activity from multi-scale C2 activity.

    :param c2s: C2 activity
    :type c2s: 2D ndarray of float
    :returns: IT activity for each prototype
    :rtype: 1D ndarray of float

    """
    it = np.array(c2s).max(0)
    return it

def Whiten(data):
  """Normalize an array, such that each location contains equal energy.

  For each X-Y location, the vector :math:`a` of data (containing activation for
  each band) is *sphered* according to:

  .. math::

    a' = (a - \mu_a ) / \sigma_a

  where :math:`\mu_a` and :math:`\sigma_a` are the mean and standard deviation
  of :math:`a`, respectively.

  .. caution::

     This function modifies the input data in-place.

  :param data: Layer activity to modify.
  :type data: 3D ndarray of float
  :returns: The `data` array.
  :rtype: 3D ndarray of float

  """
  data -= data.mean(0)
  norms = np.sqrt((data**2).sum(0))
  norms[ norms < 1 ] = 1
  data /= norms
  return data
