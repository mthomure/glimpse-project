"""This module implements a two-stage HMAX model.

This module implements a multi-scale analysis by applying Gabors corresponding
to different edge widths. The model given here corresponds as closely as
possible to the configuration used by Serre et al (2007).

"""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import numpy as np

from glimpse.models.misc import BaseLayer, LayerSpec, BaseState, BaseModel
from glimpse.util import ACTIVATION_DTYPE
from glimpse.util import docstring
from glimpse.util.garray import CropArray
from .params import Params, S1Params

class Layer(BaseLayer):

  #: Specifier for the result of S1 filtering.
  S1 = LayerSpec("s1", "S1", BaseLayer.IMAGE)

  #: Specifier for the result of C1 pooling.
  C1 = LayerSpec("c1", "C1", S1)

  #: Specifier for the result of S2 filtering.
  S2 = LayerSpec("s2", "S2", C1)

  #: Specifier for the result of C2 (global) pooling.
  C2 = LayerSpec("c2", "C2", S2)

class State(BaseState):
  """A container for the :class:`Model` state."""
  pass

class Model(BaseModel):
  """A two-stage, HMAX hierarchy of S+C layers."""

  #: The datatype associated with layer descriptors for this model.
  LayerClass = Layer

  #: The parameters type associated with this model.
  ParamClass = Params

  #: The datatype associated with network states for this model.
  StateClass = State

  @docstring.copy(BaseModel.__init__)
  def __init__(self, backend = None, params = None):
    super(Model, self).__init__(backend, params)

    # Stores the kernels for the S2 layer.
    self._s2_kernels = None

  @property
  def num_orientations(self):
    """The number of S1 orientations used.

    :rtype: int

    """
    return S1Params.num_orientations

  @property
  def num_scale_bands(self):
    """The number of V1 scale bands.

    This is equal to the number of C1 and S2 scales, but not the number of S1
    scales.

    :rtype: int

    """
    return len(self.params.v1)

  @property
  def s1_kernels_are_normed(self):
    """Determine if the model uses unit-norm S1 kernels."""
    return False

  @property
  def s2_kernel_shapes(self):
    """The expected shape of the S2 kernels, *without* band structure.

    :rtype: 2D tuple of int

    """
    return tuple( (self.num_orientations, kw, kw)
        for kw in self.params.s2_kwidths )

  @property
  def s2_kernel_sizes(self):
    """The set of supported S2 kernel sizes.

    :rtype: tuple of int

    """
    return tuple(self.params.s2_kwidths)

  @property
  def s2_kernels_are_normed(self):
    """Determine if the model uses unit-norm S2 kernels."""
    return False

  @property
  def s2_kernels(self):
    """The S2 kernels array.

    :returns: S2 kernels indexed by kernel size, kernel offset, orientation.
    :rtype: list of 4D ndarray of float

    """
    return self._s2_kernels

  @s2_kernels.setter
  def s2_kernels(self, kernels):
    """Set the kernels for the S2 layer.

    :param kernels: S2 prototype kernels indexed by kernel size, offset,
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
    if output_id == L.S1.ident:
      return self.BuildS1FromImage(state[L.IMAGE.ident])
    elif output_id == L.C1.ident:
      return self.BuildC1FromS1(state[L.S1.ident])
    elif output_id == L.S2.ident:
      return self.BuildS2FromC1(state[L.C1.ident])
    elif output_id == L.C2.ident:
      return self.BuildC2FromS2(state[L.S2.ident])
    return super(Model, self)._BuildSingleNode(output_id, state)

  def BuildS1FromImage(self, image):
    """Apply S1 processing to image layer data.

    :param image: Result of retinal layer processing.
    :type image: 2D ndarray of float
    :returns: S1 maps indexed by scale band, filter offset, and orientation.
    :rtype: list of 4D ndarray of float

    """
    # Reshape image to be 3D array
    image = np.expand_dims(image, axis = 0)
    # List of 4-D arrays, where list index is scale band. Arrays are indexed by
    # offset within scale band, orientation, y-offset, x-offset.
    s1_bands = []
    for band_params in self.params.v1:  # loop over scale bands
      # List of 3-D arrays, where list index is offset within scale band. Arrays
      # are indexed by orientation, y-offset, x-offset.
      s1_maps_for_band = []
      for s1_params in band_params.s1_params:  # loop over filters in scale band
        kernels = s1_params.kernels
        # Reshape each kernel to be 3-D (index, 1, y, x)
        kernels = kernels.reshape((-1, 1) + kernels.shape[-2:])
        # Filter the image. Result is 3-D, where the first index corresponds to
        # the filter orientation.
        s1 = self.backend.NormDotProduct(image, kernels, bias = 1.0,
            scaling = 1)
        s1_maps_for_band.append(s1)
      # Crop to give all maps in this band the same spatial extent. The largest
      # kernel has the smallest map.
      min_height, min_width = s1_maps_for_band[-1].shape[-2:]
      s1_band = [ CropArray(s1, (min_height, min_width))
          for s1 in s1_maps_for_band ]
      # Concatenate all arrays for this scale band.
      s1_band = np.array(s1_band, dtype = ACTIVATION_DTYPE)
      s1_bands.append(s1_band)
    return s1_bands

  def BuildC1FromS1(self, s1s):
    """Compute the C1 layer activity from multi-scale S1 activity.

    :param s1s: S1 maps indexed by scale band, filter offset, and orientation.
    :type s1s: list of 4D ndarray of float
    :returns: C1 maps indexed by scale band and orientation.
    :rtype: list of 3D ndarray of float

    """
    assert len(s1s) == len(self.params.v1)
    c1_bands = []
    for band_params, s1_band in zip(self.params.v1, s1s):
      # Pool over all maps in this scale band.
      c1_band = s1_band.max(0)
      c1_band = self.backend.LocalMax(c1_band, band_params.c1_kwidth,
          scaling = band_params.c1_scaling)
      c1_bands.append(c1_band)
    return c1_bands

  def BuildS2FromC1(self, c1s):
    """Compute the S2 layer activity from multi-scale C1 activity.

    The set of activation maps for all kernel sizes are concatenated together,
    so there is one list of maps for each scale band.

    :param c1s: C1 maps indexed by scale band and orientation.
    :type c1s: list of 3D ndarray of float
    :returns: S2 maps indexed by scale band and prototype. Results for all
       kernel sizes are concatenated.
    :rtype: list of 3D ndarray of float

    """
    if self.s2_kernels == None:
      raise Exception("Need S2 kernels to compute S2 layer activity, but none "
          "were specified.")
    if len(c1s) == 0:
      return []
    # List of 3-D array indexed by scale band and prototype.
    s2_bands = []
    for c1_band in c1s:
      # Filter the input array. Result is list of 3-D array indexed by kernel
      # size and prototype offset for given size.
      s2_band = [ self.backend.Rbf(c1_band, ks, self.params.s2_beta,
          scaling = 1) for ks in self.s2_kernels ]  # loop over kernel size
      # Crop to give all maps in this band the same spatial extent. The largest
      # kernel has the smallest output map.
      min_height, min_width = s2_band[-1].shape[-2:]
      s2_band = [ CropArray(s2, (min_height, min_width)) for s2 in s2_band ]
      # Concatenate results across kernel size. Result is 4-D array indexed by
      # kernel size and prototype offset for given size.
      s2_band = np.array(s2_band, dtype = ACTIVATION_DTYPE)
      # Concatenate activation maps for different kernel sizes. Result is 3-D
      # array indexed by prototype.
      s2_band = s2_band.reshape((-1,) + s2_band.shape[-2:])
      s2_bands.append(s2_band)
    return s2_bands

  def BuildC2FromS2(self, s2s):
    """Compute global C2 layer activity from multi-scale S2 activity.

    :param s2s: S2 maps indexed by scale band and prototype.
    :type: s2s: list of 3D ndarray of float
    :returns: C2 activity for each prototype.
    :rtype: 1D ndarray of float

    """
    # Pool over location.
    c2s = map(self.backend.GlobalMax, s2s)
    c2s = np.array(c2s, ACTIVATION_DTYPE)
    # Pool over scale.
    c2s = c2s.max(0)
    return c2s

# Add (circular) Model reference to State class.
State.ModelClass = Model
