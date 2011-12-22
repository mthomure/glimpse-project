# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.models.viz2.ops import ModelOps as BaseModelOps
from glimpse.models.viz2.ops import Whiten
from glimpse.util import kernel
from params import Params
from scipy.ndimage.interpolation import zoom

class ModelOps(BaseModelOps):
  """Base class for a Glimpse model based on Mutch & Lowe (2008). This class
  implements all single-layer transformations."""

  # The parameters type associated with this model.
  Params = Params

  @property
  def s1_kernel_shape(self):
    """Get the expected shape of the S1 kernels array (i.e., includes band
    structure).
    RETURN (tuple) expected shape"""
    p = self.params
    return p.s1_num_orientations, p.s1_num_phases, p.s1_kwidth, p.s1_kwidth

  @property
  def s1_kernels(self):
    """Get the S1 kernels array, generating it if unset."""
    # if kernels array is empty, then generate it using current model parameters
    if self._s1_kernels == None:
      p = self.params
      self._s1_kernels = kernel.MakeGaborKernels(
          kwidth = p.s1_kwidth,
          num_orientations = p.s1_num_orientations,
          num_phases = p.s1_num_phases, shift_orientations = True,
          scale_norm = True)
    return self._s1_kernels

  def BuildS1FromRetina(self, retina):
    """Apply S1 processing to some existing retinal layer data.
    retina -- (2-D array) result of retinal layer processing
    RETURNS list of (4-D) S1 activity arrays, with one array per scale
    """
    # Create scale pyramid of retinal map
    p = self.params
    retina_scales = [ zoom(retina, 1 / p.scale_factor ** scale)
        for scale in range(p.num_scales) ]
    # Reshape kernel array to be 3-D: index, 1, y, x
    s1_kernels = self.s1_kernels.reshape((-1, 1, p.s1_kwidth, p.s1_kwidth))
    s1s = []
    for scale in range(p.num_scales):
      # Reshape retina to be 3D array
      retina = retina_scales[scale]
      retina_ = retina.reshape((1,) + retina.shape)
      s1_ = self.backend.NormRbf(retina_, s1_kernels, bias = p.s1_bias,
          beta = p.s1_beta, scaling = p.s1_scaling)
      # Reshape S1 to be 4D array
      s1 = s1_.reshape((p.s1_num_orientations, p.s1_num_phases) + \
          s1_.shape[-2:])
      # Pool over phase.
      s1 = s1.max(1)
      s1s.append(s1)
    return s1s

  def BuildC1FromS1(self, s1s):
    p = self.params
    c1s = [ self.backend.LocalMax(s1, kwidth = p.c1_kwidth,
        scaling = p.c1_scaling) for s1 in s1s ]
    if p.c1_whiten:
      map(Whiten, c1s)
    return c1s

  def BuildS2FromC1(self, c1s):
    if self.s2_kernels == None:
      raise Exception("Need S2 kernels to compute S2 layer activity, but none "
          "were specified.")
    p = self.params
    s2s = []
    for scale in range(p.num_scales):
      c1 = c1s[scale]
      s2 = self.backend.NormRbf(c1, self.s2_kernels, bias = p.s2_bias,
          beta = p.s2_beta, scaling = p.s2_scaling)
      s2s.append(s2)
    return s2s
