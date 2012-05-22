"""This module implements a two-stage HMAX-like model.

This module implements a multi-scale analysis by applying Gabors corresponding
to different edge widths. The model given here was used for the GCNC 2011
experiments.

"""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import numpy as np

from glimpse.models.misc import BaseLayer, LayerSpec, BaseState, BaseModel, \
    Whiten
from glimpse.util import ACTIVATION_DTYPE
from glimpse.util import kernel
from glimpse.util import docstring
from .params import Params

class Layer(BaseLayer):
  """Enumerator for model layers."""

  #: Specifier for the preprocessed input.
  RETINA = LayerSpec("r", "retina", BaseLayer.IMAGE)

  #: Specifier for the result of S1 filtering.
  S1 = LayerSpec("s1", "S1", RETINA)

  #: Specifier for the result of C1 pooling.
  C1 = LayerSpec("c1", "C1", S1)

  #: Specifier for the result of S2 filtering.
  S2 = LayerSpec("s2", "S2", C1)

  #: Specifier for the result of C2 (local) pooling.
  C2 = LayerSpec("c2", "C2", S2)

class State(BaseState):
  """A container for the :class:`Model` state."""
  pass

class Model(BaseModel):
  """A two-stage, HMAX-like hierarchy of S+C layers."""

  #: The datatype associated with layer descriptors for this model.
  LayerClass = Layer

  #: The parameters type associated with this model.
  ParamClass = Params

  #: The datatype associated with network states for this model.
  StateClass = State

  @docstring.copy(BaseModel.__init__)
  def __init__(self, backend = None, params = None):
    super(Model, self).__init__(backend, params)
    # (5D ndarray of float) S1 kernels indexed by scale, orientation, and phase.
    self._s1_kernels = None
    # (list of 3D ndarray of float) S2 kernels indexed by scale and orientation.
    self._s2_kernels = None

  @property
  def s1_kernels_are_normed(self):
    """Determine if the model uses unit-norm S1 kernels."""
    return self.params.s1_operation in ('NormDotProduct', 'NormRbf')

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
    """The set of S1 kernels, which is generated if not set.

    :returns: S1 kernels indexed by scale, orientation, and phase.
    :rtype: 5D ndarray of float

    """
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
    """Set the kernels for the S1 layer.

    :param kernels: S1 kernels indexed by scale, orientation, and phase.
    :type kernels: 5D ndarray of float

    """
    if kernels == None:
      self._s1_kernels = None
      return
    if kernels.shape != self.s1_kernel_shape:
      raise ValueError("S1 kernels have wrong shape: expected %s, got %s" % \
          (self.s1_kernel_shape, kernels.shape))
    if self.s1_kernels_are_normed:
      if not np.allclose(np.array(map(np.linalg.norm, kernels)), 1):
        raise ValueError("S1 kernels are not normalized")
    if np.isnan(kernels).any():
      raise ValueError("S1 kernels contain NaN")
    self._s1_kernels = kernels.astype(ACTIVATION_DTYPE)

  @property
  def s2_kernel_shapes(self):
    """The expected shape of a single S2 kernel, *without* band structure.

    :rtype: tuple of int

    """
    p = self.params
    kshape = p.s1_num_orientations, p.s2_kwidth, p.s2_kwidth
    return (kshape,)

  @property
  def s2_kernel_sizes(self):
    """The set of supported S2 kernel sizes, given as a tuple of int."""
    return (self.params.s2_kwidth,)

  @property
  def s2_kernels_are_normed(self):
    """Determine if the model uses unit-norm S2 kernels."""
    return self.params.s2_operation in ('NormDotProduct', 'NormRbf')

  @property
  def s2_kernels(self):
    """The set of S2 kernels.

    :returns: S2 kernels indexed by kernel size, kernel offset, and orientation.
    :rtype: list of 4D ndarray of float

    """
    return self._s2_kernels

  @s2_kernels.setter
  def s2_kernels(self, kernels):
    """Set the kernels for the S2 layer.

    :param kernels: S2 prototype kernels indexed by kernel size, offset, and
       orientation.
    :type kernels: list of 4D ndarray of float

    """
    if kernels == None:
      self._s2_kernels = None
      return
    # Check kernel shapes
    num_sizes = len(self.s2_kernel_shapes)
    # Kernels should be a list of arrays, with one array per S2 kernel size.
    if len(kernels) != num_sizes:
      raise ValueError("Expected S2 kernels of %d sizes, got %d sizes" % \
          (num_sizes, len(kernels)))
    for expected_kshape, ks in zip(self.s2_kernel_shapes, kernels):
      # Kernels should be square (spatially), with the correct number of bands.
      if expected_kshape != ks.shape[1:]:
        raise ValueError("S2 kernels have wrong shape: expected "
        " (*, %d, %d, %d), got %s" % (expected_kshape + (ks.shape,)))
      # Kernels should have only real values.
      if np.isnan(ks).any():
        raise ValueError("S2 kernels contain NaN")
    self._s2_kernels = [ ks.astype(ACTIVATION_DTYPE) for ks in kernels ]

  @docstring.copy(BaseModel._BuildSingleNode)
  def _BuildSingleNode(self, output_id, state):
    L = self.LayerClass
    if output_id == L.RETINA.ident:
      return self.BuildRetinaFromImage(state[L.IMAGE.ident])
    elif output_id == L.S1.ident:
      return self.BuildS1FromRetina(state[L.RETINA.ident])
    elif output_id == L.C1.ident:
      return self.BuildC1FromS1(state[L.S1.ident])
    elif output_id == L.S2.ident:
      return self.BuildS2FromC1(state[L.C1.ident])
    elif output_id == L.C2.ident:
      return self.BuildC2FromS2(state[L.S2.ident])
    return super(Model, self)._BuildSingleNode(output_id, state)

  def BuildRetinaFromImage(self, img):
    """Compute retinal layer activity from the input image.

    :param img: Image data.
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
    """Compute the S1 layer activity from retinal layer activity.

    .. note::

       This method pools over phase, so the output array has only scale and
       orientation bands.

    :param retina: Result of retinal layer processing.
    :type retina: 2D ndarray of float
    :returns: S1 maps indexed by scale and orientation.
    :rtype: 4D ndarray of float

    """
    # Reshape retina to be 3D array
    p = self.params
    retina_ = retina.reshape((1,) + retina.shape)
    # Reshape kernel array to be 4-D (scale, index, 1, y, x)
    s1_kernels = self.s1_kernels.reshape((-1, 1, p.s1_kwidth, p.s1_kwidth))
    backend_op = getattr(self.backend, p.s1_operation)
    s1_ = backend_op(retina_, s1_kernels, bias = p.s1_bias, beta = p.s1_beta,
        scaling = p.s1_sampling)
    # Reshape S1 to be 5D array
    s1 = s1_.reshape((p.num_scales, p.s1_num_orientations, p.s1_num_phases) + \
        s1_.shape[-2:])
    # Pool over phase.
    s1 = s1.max(2)
    return s1

  def BuildC1FromS1(self, s1s):
    """Compute the C1 layer activity from multi-scale S1 activity.

    :param s1s: S1 maps indexed by scale and orientation.
    :type s1s: 4D ndarray of float
    :returns: C1 maps indexed by scale and index.
    :rtype: 4D ndarray of float

    """
    p = self.params
    c1s = [ self.backend.LocalMax(s1, kwidth = p.c1_kwidth,
        scaling = p.c1_sampling) for s1 in s1s ]
    c1s = np.array(c1s, dtype = ACTIVATION_DTYPE)
    if p.c1_whiten:
      #~ # DEBUG: use this to whiten over orientation only
      #~ map(Whiten, c1s)
      # DEBUG: use this to whiten over scale AND orientation concurrently. PANN
      # and the old Viz2 model used this.
      c1_shape = c1s.shape
      c1s = c1s.reshape((-1,) + c1s.shape[-2:])
      Whiten(c1s)
      c1s = c1s.reshape(c1_shape)
    return c1s

  def BuildS2FromC1(self, c1s):
    """Compute the S2 layer activity from multi-scale C1 activity.

    :param c1s: C1 maps indexed by scale and orientation.
    :type c1s: 4D ndarray of float, or list of 3D ndarray of float
    :returns: S2 maps indexed by scale and prototype. Results for all kernel
       sizes are concatenated.
    :rtype: 4D ndarray of float

    """
    if self.s2_kernels == None or len(self.s2_kernels[0]) == 0:
      raise Exception("Need S2 kernels to compute S2 layer activity, but none "
          "were specified.")
    kernels = self.s2_kernels[0]
    if len(c1s) == 0:
      return []
    p = self.params
    c1s = np.array(c1s, copy = False)
    # Get the spatial extent of a map. Note that this assumes that every C1 map
    # has same spatial extent.
    height, width = c1s[0].shape[-2:]
    # Get the shape of each S2 map
    s2_shape = self.backend.OutputMapShapeForInput(p.s2_kwidth, p.s2_kwidth,
        p.s2_sampling, height, width)
    # Get the shape of the full S2 activity array
    s2_shape = (p.num_scales, len(kernels)) + s2_shape
    s2s = np.empty(s2_shape, ACTIVATION_DTYPE)
    backend_op = getattr(self.backend, p.s2_operation)
    for scale in range(p.num_scales):
      c1 = c1s[scale]
      s2 = s2s[scale]
      backend_op(c1, kernels, bias = p.s2_bias, beta = p.s2_beta,
          scaling = p.s2_sampling, out = s2)
    return s2s

  def BuildC2FromS2(self, s2s):
    """Compute global C2 layer activity from multi-scale S2 activity.

    :param s2s: S2 maps indexed by scale and prototype.
    :type: s2s: 4D ndarray of float
    :returns: C2 activity for each prototype.
    :rtype: 1D ndarray of float

    """
    # Pool over location, resulting in a list of 1D arrays.
    c2s = map(self.backend.GlobalMax, s2s)
    # Concatenate arrays, resulting in a single 2D array.
    c2s = np.array(c2s, ACTIVATION_DTYPE)
    # Pool over scale, resulting in a 1D array.
    c2s = c2s.max(0)
    return c2s

# Add (circular) Model reference to State class.
State.ModelClass = Model
