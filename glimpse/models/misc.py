# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# General functions that are applicable to multiple models.

from glimpse.util import ImageToArray
import Image
import numpy as np
import random

def ImageLayerFromInputArray(input_, backend):
  """Create the initial image layer from some input.
  input_ -- Image or (2-D) array of input data. If array, values should lie in
            the range [0, 1].
  RETURNS (2-D) array containing image layer data
  """
  if isinstance(input_, Image.Image):
    input_ = input_.convert('L')
    input_ = ImageToArray(input_, transpose = True)
    input_ = input_.astype(np.float)
    # Map from [0, 255] to [0, 1]
    input_ /= 255
  return backend.PrepareArray(input_)

def SampleC1Patches(c1s, kwidth):
  """
  Sample patches from a layer of C1 activity.
  c1s - (4-D array, or list of 3-D arrays) C1 activity maps, one map per scale.
  RETURNS: infinite iterator over prototype arrays and corresponding locations
           (i.e., iterator elements are 2-tuples). Prototype location gives
           top-left corner of C1 region.
  """
  assert (np.array(len(c1s[0].shape)) == 3).all()
  num_scales = len(c1s)
  num_orientations = len(c1s[0])
  while True:
    scale = random.randint(0, num_scales - 1)
    c1 = c1s[scale]
    height, width = c1.shape[-2:]
    y = random.randint(0, height - kwidth)
    x = random.randint(0, width - kwidth)
    patch = c1[ :, y : y + kwidth, x : x + kwidth ]
    yield patch.copy(), (scale, y, x)
