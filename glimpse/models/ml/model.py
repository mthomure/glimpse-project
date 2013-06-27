"""Defines a multi-layer, HMAX-like model.

This module implements a multi-scale analysis by applying single-scale Gabors to
a scale pyramid of the input image. This is similar to the configuration used by
Mutch & Lowe (2008).

"""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import numpy as np

from glimpse.backends import BackendError, ACTIVATION_DTYPE
from glimpse.util.kernel import MakeGaborKernels
from glimpse.util.gimage import MakeScalePyramid

from glimpse.models.base.model import layer_builder, Layer as BaseLayer, \
    Model as BaseModel, State as BaseState
from glimpse.models.base.layer import LayerSpec
from .misc import *
from .param import Params, SLayerOp

__all__ = [
    'Layer',
    'State',
    'Model',
    ]

class Layer(BaseLayer):
  """Enumerator for model layers."""

  #: Specifier for the preprocessed input.
  RETINA = LayerSpec("r", "retina", [BaseLayer.IMAGE])

  #: Specifier for the result of S1 filtering.
  S1 = LayerSpec("s1", "S1", [RETINA])

  #: Specifier for the result of C1 pooling.
  C1 = LayerSpec("c1", "C1", [S1])

  #: Specifier for the result of S2 filtering.
  S2 = LayerSpec("s2", "S2", [C1])

  #: Specifier for the result of C2 (local) pooling.
  C2 = LayerSpec("c2", "C2", [S2])

class State(BaseState):
  """A container for the :class:`Model` state."""
  pass

class Model(BaseModel):
  """Create a 2-part, HMAX-like hierarchy of S+C layers."""

  #: The datatype associated with layer descriptors for this model.
  LayerClass = Layer

  #: The parameters type associated with this model.
  ParamClass = Params

  #: The datatype associated with network states for this model.
  StateClass = State

  #: (5D ndarray of float) S1 kernels indexed by scale, orientation, and phase.
  _s1_kernels = None

  #: (list of 3D ndarray of float) S2 kernels indexed by scale and orientation.
  _s2_kernels = None

  @property
  def s1_kernels(self):
    """The set of S1 kernels, which is generated if not set.

    :returns: S1 kernels indexed by orientation, and phase.
    :rtype: 4D ndarray of float

    """
    # if kernels array is empty, then generate it using current model parameters
    if self._s1_kernels == None:
      p = self.params
      self._s1_kernels = MakeGaborKernels(
          kwidth = p.s1_kwidth,
          num_orientations = p.s1_num_orientations,
          num_phases = p.s1_num_phases, shift_orientations = True,
          scale_norm = p.s1_kernels_are_normed).astype(ACTIVATION_DTYPE)
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
    p = self.params
    if kernels.shape != p.s1_kernel_shape:
      raise ValueError("S1 kernels have wrong shape: expected %s, got %s" % \
          (p.s1_kernel_shape, kernels.shape))
    if p.s1_kernels_are_normed:
      if not np.allclose(np.array(map(np.linalg.norm, kernels.reshape((-1,) +
          kernels.shape[-2:]))), 1):
        raise ValueError("S1 kernels are not normalized")
    if np.isnan(kernels).any():
      raise ValueError("S1 kernels contain NaN")
    self._s1_kernels = kernels.astype(ACTIVATION_DTYPE)

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
    num_sizes = len(self.params.s2_kernel_shapes)
    # Kernels should be a list of arrays, with one array per S2 kernel size.
    if len(kernels) != num_sizes:
      raise ValueError("Expected S2 kernels of %d sizes, got %d sizes" % \
          (num_sizes, len(kernels)))
    for expected_kshape, ks in zip(self.params.s2_kernel_shapes, kernels):
      # Kernels should be square (spatially), with the correct number of bands.
      if expected_kshape != ks.shape[1:]:
        raise ValueError("S2 kernels have wrong shape: expected "
        " (*, %d, %d, %d), got %s" % (expected_kshape + (ks.shape,)))
      # Kernels should have only real values.
      if np.isnan(ks).any():
        raise ValueError("S2 kernels contain NaN")
    self._s2_kernels = [ ks.astype(ACTIVATION_DTYPE) for ks in kernels ]

  @layer_builder(Layer.RETINA)
  def BuildRetina(self, img):
    """Compute retinal layer activity from the input image.

    :param img: Image data.
    :type: 2D ndarray of float
    :rtype: 2D ndarray of float

    """
    p = self.params
    if not p.retina_enabled:
      return img
    try:
      retina = self.backend.ContrastEnhance(img, kwidth = p.retina_kwidth,
          bias = p.retina_bias, scaling = 1)
    except BackendError, ex:
      ex.layer = Layer.RETINA
      raise
    return retina

  @layer_builder(Layer.S1)
  def BuildS1(self, retina):
    """Apply S1 processing to some existing retinal layer data.

    .. note::

       This method pools over phase, so the output has only scale and
       orientation bands.

    :param retina: Result of retinal layer processing.
    :type retina: 2D ndarray of float
    :return: S1 maps indexed by scale and orientation.
    :rtype: list of 3D ndarray of float

    """
    # Create scale pyramid of retinal map
    p = self.params
    num_scales = p.num_scales
    if num_scales == 0:
      num_scales = NumScalesSupported(p, min(retina.shape))
    retina_pyramid = MakeScalePyramid(retina, num_scales, 1.0 / p.scale_factor)
    s1_kernels = self.s1_kernels
    if p.s1_operation == SLayerOp.NORM_DOT_PRODUCT:
      ndp = True
      # NDP is already phase invariant, just use one phase of filters
      s1_kernels = s1_kernels[:, 0].copy()
    else:
      ndp = False
    # Reshape kernel array to be 3-D: index, 1, y, x
    s1_kernels = s1_kernels.reshape((-1, 1, p.s1_kwidth, p.s1_kwidth))
    backend_op = getattr(self.backend, p.s1_operation)
    s1s = list()
    for scale in range(num_scales):
      # Reshape retina to be 3D array
      retina = retina_pyramid[scale].reshape((1,) + retina_pyramid[scale].shape)
      try:
        s1 = backend_op(retina, s1_kernels, bias = p.s1_bias, beta = p.s1_beta,
            scaling = p.s1_sampling)
      except BackendError, ex:
        ex.layer = Layer.S1
        ex.scale = scale
        raise
      if ndp:
        # Take the element-wise absolute value (in-place). S1 will now be a 3D
        # array of phase-invariant responses.
        np.abs(s1, s1)
      else:
        # Reshape S1 to be 4D array
        s1 = s1.reshape((p.s1_num_orientations, p.s1_num_phases) +
            s1.shape[-2:])
        # Pool over phase.
        s1 = s1.max(1)
      # Append 3D array to list
      s1s.append(s1)
    return s1s

  @layer_builder(Layer.C1)
  def BuildC1(self, s1s):
    """Compute the C1 layer activity from multi-scale S1 activity.

    :param s1s: S1 maps indexed by scale.
    :type s1s: list of 3D ndarray of float
    :returns: C1 maps indexed by scale and orientation.
    :rtype: list of 3D ndarray of float

    """
    p = self.params
    c1s = list()
    for scale in range(len(s1s)):
      try:
        c1s.append(self.backend.LocalMax(s1s[scale], kwidth = p.c1_kwidth,
          scaling = p.c1_sampling))
      except BackendError, ex:
        ex.layer = Layer.C1
        ex.scale = scale
        raise
    if p.c1_whiten:
      # Whiten each scale independently, modifying values in-place.
      map(Whiten, c1s)
    return c1s

  @layer_builder(Layer.S2)
  def BuildS2(self, c1s):
    """Compute the S2 layer activity from multi-scale C1 activity.

    :param c1s: C1 maps indexed by scale and orientation.
    :type c1s: list of 3D ndarray of float
    :returns: S2 maps indexed by scale, kernel width, and prototype.
    :rtype: list of list of 3D ndarray of float

    """
    if self.s2_kernels == None or len(self.s2_kernels[0]) == 0:
      raise Exception("Need S2 kernels to compute S2 layer activity, but none "
          "were specified.")
    if len(c1s) == 0:
      return list()
    p = self.params
    backend_op = getattr(self.backend, p.s2_operation)
    s2s = list()
    for scale, c1 in enumerate(c1s):
      s2_by_kwidth = list()
      for ks in self.s2_kernels:
        try:
          s2 = backend_op(c1, ks, bias = p.s2_bias, beta = p.s2_beta,
              scaling = p.s2_sampling)
        except BackendError, ex:
          # Annotate exception with scale information.
          ex.layer = Layer.S2
          ex.scale = scale
          raise
        # Treat 3D array as list of 2D arrays, and contatenate existing list.
        s2_by_kwidth.append(s2)
      # Append list of 2D array to list.
      s2s.append(s2_by_kwidth)
    return s2s

  @layer_builder(Layer.C2)
  def BuildC2(self, s2s):
    """Compute global C2 layer activity from multi-scale S2 activity.

    :param s2s: S2 maps indexed by scale and prototype.
    :type: s2s: list of list of 2D ndarray of float
    :returns: C2 activity for each prototype.
    :rtype: 1D ndarray of float

    """
    c2s = list()  # 2D list of 1D arrays, indexed by (scale,kwidth)
    for scale, s2_by_kwidth in enumerate(s2s):
      c2_by_kwidth = list()
      for s2 in s2_by_kwidth:
        try:
          # Pool over location, resulting in a 1D array.
          c2_by_kwidth.append(self.backend.GlobalMax(s2))
        except BackendError, ex:
          ex.layer = Layer.C2
          ex.scale = scale
          raise
      c2s.append(c2_by_kwidth)
    c2s = zip(*c2s)  # reorder list by (kwidth,scale)
    # Pool over scale, resuling in a list of 1D array.
    c2s = [ np.array(c2_by_scale).max(0) for c2_by_scale in c2s ]
    return c2s

# Add (circular) Model reference to State class.
State.ModelClass = Model
