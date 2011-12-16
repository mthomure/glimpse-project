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
  """Base class for a Glimpse model based on PANN. This class implements all
  single-layer transformations."""

  # The parameters type associated with this model.
  Params = Params

  def __init__(self, backend = None, params = None):
    """Create new object.
    backend -- (implements IBackend) implementation of backend operations, such
               as dot-products
    params -- (Params) model configuration, such as S-unit kernel widths
    """
    if backend == None:
      backend = backends.MakeBackend()
    else:
      backend = copy.copy(backend)
    if params == None:
      params = self.Params()
    else:
      if not isinstance(params, self.Params):
        raise ValueError("Params object has wrong type: expected %s, got %s" % \
            (self.Params, type(params)))
      params = copy.copy(params)
    self.backend = backend
    self.params = params
    self._s1_kernels = None
    self._s2_kernels = None

  @property
  def s1_kernel_shape(self):
    """Get the expected shape of the S1 kernels array (i.e., includes band
    structure).
    RETURN (tuple) expected shape
    """
    p = self.params
    return p.num_scales, p.s1_num_orientations, p.s1_num_phases, p.s1_kwidth, \
        p.s1_kwidth

  @property
  def s1_kernels(self):
    """Get the S1 kernels array, generating it if unset."""
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
    else:
      self._s1_kernels = kernels.astype(ACTIVATION_DTYPE)

  @property
  def s2_kernel_shape(self):
    """Get the expected shape of a single S2 kernel (i.e., this does not include
    band structure).
    RETURN (tuple) expected shape
    """
    p = self.params
    return p.s1_num_orientations, p.s2_kwidth, p.s2_kwidth

  @property
  def s2_kernels(self):
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
    if not np.allclose(np.array(map(np.linalg.norm, kernels)), 1):
      raise ValueError("S2 kernels are not normalized")
    if np.isnan(kernels).any():
      raise ValueError("S2 kernels contain NaN")
    self._s2_kernels = kernels.astype(ACTIVATION_DTYPE)

  def BuildImageFromInput(self, input_):
    """Create the initial image layer from some input.
    input_ -- Image or (2-D) array of input data. If array, values should lie in
              the range [0, 1].
    RETURNS (2-D) array containing image layer data
    """
    return ImageLayerFromInputArray(input_, self.backend)

  def BuildRetinaFromImage(self, img):
    p = self.params
    if not p.retina_enabled:
      return img
    retina = self.backend.ContrastEnhance(img, kwidth = p.retina_kwidth,
        bias = p.retina_bias, scaling = 1)
    return retina

  def BuildS1FromRetina(self, retina):
    """Apply S1 processing to some existing retinal layer data.
    retina -- (2-D array) result of retinal layer processing
    RETURNS list of (4-D) S1 activity arrays, with one array per scale
    """
    # Reshape retina to be 3D array
    p = self.params
    retina_ = retina.reshape((1,) + retina.shape)
    # Reshape kernel array to be 4-D: scale, index, 1, y, x
    s1_kernels = self.s1_kernels.reshape((-1, 1, p.s1_kwidth, p.s1_kwidth))
    s1_ = self.backend.NormRbf(retina_, s1_kernels, bias = p.s1_bias,
        beta = p.s1_beta, scaling = p.s1_scaling)
    # Reshape S1 to be 5D array
    s1 = s1_.reshape((p.num_scales, p.s1_num_orientations, p.s1_num_phases) + \
        s1_.shape[-2:])
    # Pool over phase.
    s1 = s1.max(2)
    return s1

  def BuildC1FromS1(self, s1s):
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
    c1s -- (4D ndarray, or list of 3D ndarrays) C1 activity
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
    for scale in range(p.num_scales):
      c1 = c1s[scale]
      s2 = s2s[scale]
      self.backend.NormRbf(c1, self.s2_kernels, bias = p.s2_bias,
          beta = p.s2_beta, scaling = p.s2_scaling, out = s2)
    return s2s

  def BuildC2FromS2(self, s2s):
    c2s = map(self.backend.GlobalMax, s2s)
    c2s = np.array(c2s, ACTIVATION_DTYPE)
    return c2s

  def BuildItFromC2(self, c2s):
    it = np.array(c2s).max(0)
    return it

def Whiten(data):
  data -= data.mean(0)
  norms = np.sqrt((data**2).sum(0))
  norms[ norms < 1 ] = 1
  data /= norms
  return data
