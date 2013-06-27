# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from glimpse.models.base import Params as BaseParams
from glimpse.util import traits

class KWidth(traits.BaseInt):
  """A trait type corresponding to a Glimpse kernel width (a positive, odd
  integer)."""

  default_value = 1

  info_text = 'a positive integer'

  def validate(self, object, name, value):
    value = super(KWidth, self).validate(object, name, value)
    if value > 0:
      return value
    self.error(object, name, value)

class KWidthList(traits.List):
  """A list of kernel widths.

  Note that a singleton list can be specified as a scalar.

  """

  def __init__(self, value=None, **kw):
    super(KWidthList, self).__init__(KWidth, value, minlen=1, **kw)

  def validate(self, object, name, value):
    if not hasattr(value, '__len__'):
      value = [value]
    else:
      value = list(value)  # should be list, not tuple
    return super(KWidthList, self).validate(object, name, value)

class SLayerOp(traits.Enum):
  """A trait type corresponding to a Glimpse S-unit filter operation."""

  info_text = 'S-unit filter type'

  DOT_PRODUCT = "DotProduct"
  NORM_DOT_PRODUCT = "NormDotProduct"
  RBF = "Rbf"
  NORM_RBF = "NormRbf"

  def __init__(self, value, **metadata):
    super(SLayerOp, self).__init__(value, (self.DOT_PRODUCT,
        self.NORM_DOT_PRODUCT, self.RBF, self.NORM_RBF), **metadata)

class OperationType(traits.Enum):
  """A trait type describing how S- and C-layer operations are applied."""

  #: Valid convolution: only compute an output value when the input window is
  #: fully defined (output map will be smaller than input map by
  #: kernel_width - 1).
  VALID = "valid"

  def __init__(self, value, **metadata):
    super(OperationType, self).__init__(value, ("valid"),
        **metadata)

class Params(BaseParams):
  """Parameter container for :mod:`glimpse.models.ml`."""

  retina_bias = traits.Range(low = 0., value = 1., label = "Retina Bias",
      desc = "term added to standard deviation of local window")
  retina_enabled = traits.Bool(False, label = "Retina Enabled",
      desc = "indicates whether the retinal layer is used")
  retina_kwidth = KWidth(15, label = "Retina Kernel Width",
      desc = "spatial width of input neighborhood for retinal units")

  s1_beta = traits.Range(low = 0., value = 1., exclude_low = True,
      label = "S1 Beta", desc = "term added to the norm of the input vector")
  s1_bias = traits.Range(low = 0., value = 0.01, label = "S1 Bias",
      desc = "beta parameter of RBF for S1 cells")
  s1_kwidth = KWidth(11, label = "S1 Kernel Width",
      desc = "spatial width of input neighborhood for S1 units")
  s1_num_orientations = traits.Range(low = 1, value = 4,
      label = "Number of Orientations",
      desc = "number of different S1 Gabor orientations")
  s1_num_phases = traits.Range(low = 1, value = 2, label = "Number of Phases",
      desc = "number of different phases for S1 Gabors. Using two phases "
          "corresponds to find a light bar on a dark background and vice versa")
  s1_sampling = traits.Range(low = 1, value = 1, label = "S1 Sampling",
      desc = "subsampling factor (e.g., setting this parameter to 2 will "
      "result in an S1 array that is half the width -- and half the height "
      "-- of the input array)")
  s1_shift_orientations = traits.Bool(False, label = "Shift Orientations",
      desc = "rotate Gabors by a small positive angle")
  s1_operation = SLayerOp("NormDotProduct", label = "S1 Operation")

  c1_kwidth = KWidth(11, label = "C1 Kernel Width",
      desc = "spatial width of input neighborhood for C1 units")
  c1_sampling = traits.Range(low = 1, value = 5, label = "C1 Sampling",
      desc = "subsampling factor (e.g., setting this parameter to 2 will "
      "result in a C1 array that is half the width -- and half the height "
      "-- of the S1 array)")
  c1_whiten = traits.Bool(False, label = "C1 Whiten",
      desc = "whether to normalize the total energy at each C1 location")

  s2_beta = traits.Range(low = 0., value = 5., exclude_low = True,
      label = "S2 Beta", desc = "beta parameter of RBF for S1 cells")
  # Default value is configured to match distribution of C1 norm under
  # whitening.
  s2_bias = traits.Range(low = 0., value = 0.1, label = "S2 Bias",
      desc = "additive term combined with input window norm")
  # XXX Editing `s2_kwidth` from a GUI seems to be broken.
  s2_kwidth = KWidthList(value=[7], label = "S2 Kernel Width",
      desc = "spatial width of input neighborhood for S2 units")
  s2_sampling = traits.Range(low = 1, value = 1, label = "S2 Sampling",
      desc = "subsampling factor (e.g., setting this parameter to 2 will "
      "result in an S2 array that is half the width -- and half the height "
      "-- of the C1 array)")
  s2_operation = SLayerOp("Rbf", label = "S2 Operation")
  s2_center = traits.Bool(False, label = "S2 Centering", desc = "whether to "
      "subtract input mean before applying activation function (assumes "
      "operation=Rbf)")

  operation_type = OperationType("valid", label = "Operation Type",
      desc = "the way in which S- and C-layers are applied")
  num_scales = traits.Range(low = 0, value = 9, label = "Number of Scales",
      desc = "number of different scale bands (set to zero to use as many as "
          "possible for a given image size)")
  scale_factor = traits.Range(low = 1., value = 2**(1/4.),
      label = "Scaling Factor",
      desc = "Image downsampling factor between scale bands (must be greater "
          "than one)")

  @property
  def s1_kernel_shape(self):
    """The expected shape of the S1 kernels array, including band structure.

    :rtype: tuple of int

    """
    return self.s1_num_orientations, self.s1_num_phases, self.s1_kwidth, \
        self.s1_kwidth

  @property
  def s1_kernels_are_normed(self):
    """Determine if the model uses unit-norm S1 kernels.

    :rtype: bool

    """
    return self.s1_operation in ('NormDotProduct', 'NormRbf')

  @property
  def s2_kernel_shapes(self):
    """The expected shape of a single S2 kernel.

    One shape tuple is returned for each S2 kernel shape supported by the model.

    :rtype: tuple of tuple of int

    """
    ntheta = self.s1_num_orientations
    return [ (ntheta, kw, kw) for kw in self.s2_kwidth ]

  @property
  def s2_kernel_widths(self):
    """The set of supported S2 kernel widths (i.e., spatial extents).

    :rtype: tuple of int

    """
    return self.s2_kwidth

  @s2_kernel_widths.setter
  def s2_kernel_widths(self, kwidths):
    """Set the supported S2 kernel widths (i.e., spatial extents).

    :param kwidths: int or tuple of int

    """
    self.s2_kwidth = kwidths

  @property
  def s2_kernels_are_normed(self):
    """Determine if the model uses unit-norm S2 kernels.

    :rtype: bool

    """
    return self.s2_operation in ('NormDotProduct', 'NormRbf')
