# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import copy
from glimpse.models.misc import ImageLayerFromInputArray, SampleC1Patches
from glimpse.util import kernel
from glimpse.util import ACTIVATION_DTYPE
import numpy as np

class ModelOps(object):
  """Base class for the Viz2 model. This class implements all single-layer
  transformations."""

  def __init__(self, backend, params):
    self._backend = copy.copy(backend)
    self._params = copy.copy(params)
    self._s1_kernels = None
    self._s2_kernels = None

  @property
  def s1_kernels(self):
    """Get the S1 kernels array, generating it if unset."""
    # if kernels array is empty, then generate it using current model parameters
    if self._s1_kernels == None:
      p = self._params
      self._s1_kernels = kernel.MakeMultiScaleGaborKernels(
        kwidth = p.s1_kwidth, num_scales = p.num_scales,
        num_orientations = p.s1_num_orientations, num_phases = p.s1_num_phases,
        shift_orientations = True, scale_norm = True)

  @s1_kernels.setter
  def s1_kernels(self, kernels):
    p = self._params
    expected_shape = (p.num_scales, p.s1_num_orientations, p.s1_num_phases,
        p.s1_kwidth, p.s1_kwidth)
    if kernels.shape != expected_shape:
      raise ValueError("S1 kernels have wrong shape: expected %s, got %s" % \
          (expected_shape, kernels.shape))
    else:
      self._s1_kernels = kernels.astype(ACTIVATION_DTYPE)

  @property
  def s2_kernels(self):
    return self._s2_kernels

  @s2_kernels.setter
  def s2_kernels(self, kernels):
    p = self._params
    expected_shape = p.s1_num_orientations, p.s2_kwidth, p.s2_kwidth
    if kernels.shape[1:] != expected_shape:
      raise ValueError("S2 kernels have wrong shape: expected "
          " (*, %s, %s, %s), got %s" % (expected_shape + \
          [kernels.shape]))
    else:
      self._s2_kernels = kernels.astype(ACTIVATION_DTYPE)

  def BuildImageFromInput(self, input_):
    """Create the initial image layer from some input.
    input_ -- Image or (2-D) array of input data. If array, values should lie in
              the range [0, 1].
    RETURNS (2-D) array containing image layer data
    """
    return ImageLayerFromInputArray(input_, self._backend)

  def BuildRetinaFromImage(self, img):
    p = self._params
    if not p.retina_enabled:
      return img
    retina = self._backend.ContrastEnhance(img, kwidth = p.retina_kwidth,
        bias = p.retina_bias, scaling = 1)
    return retina

  def BuildS1FromRetina(self, retina):
    """Apply S1 processing to some existing retinal layer data.
    retina -- (2-D array) result of retinal layer processing
    RETURNS list of (4-D) S1 activity arrays, with one array per scale
    """
    # Reshape retina to be 3D array
    p = self._params
    retina_ = retina.reshape((1,) + retina.shape)
    # Reshape kernel array to be 4-D: scale, index, 1, y, x
    s1_kernels = self.s1_kernels.reshape((p.num_scales, -1, 1, p.s1_kwidth,
        p.s1_kwidth))
    s1s = []
    for scale in range(p.num_scales):
      ks = s1_kernels[scale]
      s1_ = self._backend.NormRbf(retina_, ks, bias = p.s1_bias,
          beta = p.s1_beta, scaling = p.s1_scaling)
      # Reshape S1 to be 4D array
      s1 = s1_.reshape((p.s1_num_orientations, p.s1_num_phases) + \
          s1_.shape[-2:])
      # Pool over phase.
      s1 = s1.max(1)
      s1s.append(s1)
    return s1s

  def BuildC1FromS1(self, s1s):
    p = self._params
    c1s = [ self._backend.LocalMax(s1, kwidth = p.c1_kwidth,
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
    if self.s2_kernels == None:
      raise Exception("Need S2 kernels to compute S2 layer activity, but none "
          "were specified.")
    CheckPrototypes(self.s2_kernels)
    p = self._params
    s2s = []
    for scale in range(p.num_scales):
      c1 = c1s[scale]
      s2 = self._backend.NormRbf(c1, self.s2_kernels, bias = p.s2_bias,
          beta = p.s2_beta, scaling = p.s2_scaling)
      s2s.append(s2)
    return s2s

  def BuildC2FromS2(self, s2s):
    c2s = map(self._backend.GlobalMax, s2s)
    return c2s

  def BuildItFromC2(self, c2s):
    it = np.array(c2s).max(0)
    return it

  def ImprintPrototypes(self, c1s, num_prototypes):
    """Compute C1 activity maps and sample patches from random locations.
    input_ -- (Image or 2-D array) unprocessed image data
    num_prototype -- (positive int) number of prototypes to imprint
    RETURNS list of prototypes, and list of corresponding locations.
    """
    proto_it = SampleC1Patches(c1s, kwidth = self._params.s2_kwidth)
    protos = list(itertools.islice(proto_it, num_prototypes))
    for proto, loc in protos:
      proto /= np.linalg.norm(proto)
    return zip(*protos)

def CheckPrototypes(prototypes):
  assert prototypes != None
  if len(prototypes.shape) == 3:
    prototypes = prototypes.reshape((1,) + prototypes.shape)
  assert np.allclose(np.array(map(np.linalg.norm, prototypes)), 1), \
      "Internal error: S2 prototypes are not normalized"
  assert not np.isnan(prototypes).any(), \
      "Internal error: found NaN in imprinted prototype."

def Whiten(data):
  data -= data.mean(0)
  norms = np.sqrt((data**2).sum(0))
  norms[ norms < 1 ] = 1
  data /= norms
  return data
