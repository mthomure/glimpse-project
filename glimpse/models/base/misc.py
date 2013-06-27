"""General functions that are applicable to multiple models."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import Image
import itertools
from math import sqrt
import numpy as np
import random

from .param import *
from glimpse.util.gimage import ScaleImage, ScaleAndCropImage
from glimpse.backends import BackendError, InputSizeError
from glimpse.util import dataflow
from glimpse.backends import ACTIVATION_DTYPE

__all__ = [
    'PrepareImage',
    'BuildLayer',
    'SamplePatches',
    'SamplePatchesFromData',
    ]

def PrepareImage(img, params):
  """Prepare an image for input into the model.

  :param Image img: Input data.
  :param params.Params params: Parameters controlling image transformation.
  :returns: Image layer data with values in the range [0, 1].
  :rtype: 2D ndarray of float

  The image may be scaled and/or cropped, depending on the settings of the
  `image_resize_method` attribute of the `params` argument. If the value is
  NONE, the image is returned unchanged. Given a value of SCALE_SHORT_EDGE,
  SCALE_LONG_EDGE, SCALE_WIDTH, or SCALE_HEIGHT, the given image edge will be
  rescaled to match `params.image_resize_length` (preserving the aspect ratio).

  Finally, the `image_resize_method` attribute could be SCALE_AND_CROP. In this
  case, the image is scaled and cropped to a fixed size of (w, w/rho), where w
  is `image_resize_length` and rho is `image_resize_aspect_ratio`. This is
  achieved scaling the short edge (preserving aspect ratio) and then cropping
  from the long edge.

  See also :func:`ScaleImage` and :func:`ScaleAndCropImage`.

  """
  if not Image.isImageType(img):
    raise ValueError("Bad input type: %s" % type(img))
  resize_method = params.image_resize_method
  if resize_method == ResizeMethod.NONE:
    return img
  resize_length = params.image_resize_length
  old_size = np.array(img.size, np.float)  # format is (width, height)
  if resize_method == ResizeMethod.SCALE_SHORT_EDGE:
    img = ScaleImage(img, old_size / min(old_size) * resize_length)
  elif resize_method == ResizeMethod.SCALE_LONG_EDGE:
    img = ScaleImage(img, old_size / max(old_size) * resize_length)
  elif resize_method == ResizeMethod.SCALE_WIDTH:
    img = ScaleImage(img, old_size / old_size[0] * resize_length)
  elif resize_method == ResizeMethod.SCALE_HEIGHT:
    img = ScaleImage(img, old_size / old_size[1] * resize_length)
  elif resize_method == ResizeMethod.SCALE_AND_CROP:
    width = resize_length
    height = width / float(params.image_resize_aspect_ratio)
    img = ScaleAndCropImage(img, (width, height))
  else:
    raise ValueError("Unknown resize method: %s" % resize_method)
  return img

def BuildLayer(model, layer, state, save_all = True):
  """Apply the model through to the given layer.

  :param layer: Output layer(s) to compute.
  :type layer: :class:`LayerSpec` or tuple of :class:`LayerSpec`
  :param state: Initial model state from which to compute the output layer, or
     input image as path or in-memory image.
  :type state: StateClass or str or Image
  :param bool save_all: Whether the resulting state should contain values for
     all computed layers in the network, or just the output layer. Note that
     source information is always preserved.
  :returns: Output state containing the given layer
  :rtype: StateClass

  Examples:

  Get the :attr:`IMAGE <Layer.IMAGE>` layer for an image.

  >>> model = Model()
  >>> input_state = model.MakeState(glab.GetExampleImage())
  >>> output_state = BuildLayer(model, Layer.IMAGE, input_state)
  >>> assert(Layer.IMAGE in output_state)

  """
  # BuildNode can throw BackendError or DependencyErrors, which we annotate
  # with source information below. An InputLoadError may also be thrown, but
  # this already contains source information.
  try:
    state = dataflow.BuildNode(model, layer, state)
  except BackendError, ex:
    ex.source = state.get(model.LayerClass.SOURCE, None)
    raise
  except dataflow.DependencyError, ex:
    if ex.node in model.LayerClass.AllLayers():
      # If the node was valid, then we must be missing the input source.
      raise ValueError("Input state is missing source information")
    else:
      raise ValueError("An unknown model layer (%s) was requested" % ex.node)
  if not save_all:
    state_ = model.StateClass()
    # Keep output layer data
    if not hasattr(layer, '__len__'):
      layer = (layer,)
    for l in layer:
      state_[l] = state[l]
    # Keep source information
    SRC = model.LayerClass.SOURCE
    if SRC in state:
      state_[SRC] = state[SRC]
    state = state_
  return state

def SamplePatches(model, layer, num_patches, patch_size, state):
  """Extract patches from the given layer for a single image.

  Patches are sampled from random locations and scales.

  :param LayerSpec layer: Layer from which to extract patches.
  :param int num_patches: Number of patches to extract.
  :param int patch_size: Spatial extent of patches.
  :param State state: Input state for which layer activity is computed.
  :returns: list of (patch, location) pairs. The location is a triple, whose
     axes correspond to the scale, y-offset, and x-offset of the patch's
     top-left corner.
  :rtype: list of (patch, location) pairs

  See also :func:`SamplePatchesFromData`.

  """
  state = BuildLayer(model, layer, state)
  data = state[layer]
  try:
    return SamplePatchesFromData(data, patch_size, num_patches)
  except InputSizeError, ex:
    # Annotate exception with source information.
    ex.source = state.get(model.LayerClass.SOURCE)
    ex.layer = layer
    raise

def SamplePatchesFromData(data, patch_width, num_patches):
  """Sample patches from a layer of activity.

  :param data: 3D activity maps, with one map per scale. Note that the smallest
     map in the data must be at least as large as the patch width.
  :type data: ND ndarray, or list of (N-1)D ndarray. Must have N >= 3.
  :param int patch_width: Spatial extent of patch.
  :rtype: pair of arrays
  :returns: Array of patches and array of corresponding locations (elements are
     3-tuples). Location given as `(s, y, x)`, where `s` is scale, and (y, x)
     gives the top-left corner of the region.

  Each location (s, y, x) is guaranteed to lie in

     0 <= s < S

     0 <= y < H - `patch_width`

     0 <= x < W - `patch_width`

  where S is the number of scales in `data`, and H and W give its spatial height
  and width, respectively.

  Throws `InputSizeError` if the input is (spatially) smaller than
  `patch_width`.

  Examples:

  Extract 50 patches of size 10x10 from a stack of 2D arrays:

  >>> shape = (4, 100, 100)
  >>> data = numpy.random.random(shape)
  >>> patches, locs = SamplePatchesFromData(data, patch_width = 10, 50)

  """
  assert len(data) > 0
  if isinstance(data, np.ndarray) and data.ndim == 2:
    data = data.reshape((1,) + data.shape)
  assert all(d.ndim > 1 for d in data), "Data should be sequence of 3D arrays."
  num_scales = len(data)
  # Check that all scales are large enough to sample from.
  for scale_idx, scale in enumerate(data):
    if scale.shape[-1] < patch_width or scale.shape[-2] < patch_width:
      raise InputSizeError("Input data is smaller (at one or more scales) than "
          "requested patch size", scale = scale_idx)
  num_bands = data[0].ndim - 2
  patch_shape = data[0].shape[:-2] + (patch_width, patch_width)
  patches = np.empty((num_patches,) + patch_shape, ACTIVATION_DTYPE)
  locs = np.empty((num_patches, 3), np.int)
  for idx in range(num_patches):
    scale_idx = random.randint(0, num_scales - 1)
    scale = data[scale_idx]
    layer_height, layer_width = scale.shape[-2:]
    # Choose the top-left corner of the region.
    y0 = random.randint(0, layer_height - patch_width)
    x0 = random.randint(0, layer_width - patch_width)
    # Copy data from all bands in the given X-Y region.
    index = [ slice(None) ] * num_bands
    index += [ slice(y0, y0 + patch_width), slice(x0, x0 + patch_width) ]
    patches[idx] = scale[ index ]
    locs[idx] = scale_idx, y0, x0
  return patches, locs
