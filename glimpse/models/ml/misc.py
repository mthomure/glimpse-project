# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import numpy as np

def MinimumRetinaSize(params):
  """Compute the smallest retinal layer that supports the given parameters.

  This function discounts the effect of scaling.

  :param params: Parameter settings for model.
  :rtype: int
  :returns: Length of smaller edge of retina.

  """
  s2_kwidth = max(params.s2_kwidth)
  if params.operation_type == "valid":
    # Must support at least one S2 unit
    c1_size = s2_kwidth
    s1_size = c1_size * params.c1_sampling + params.c1_kwidth - 1
    retina_size = s1_size * params.s1_sampling + params.s1_kwidth - 1
  else:  # centered convolution
    c1_size = (s2_kwidth - 1) / 2
    s1_size = c1_size * params.c1_sampling + (params.c1_kwidth - 1) / 2
    retina_size = s1_size * params.s1_sampling + (params.s1_kwidth - 1) / 2
  return retina_size

def NumScalesSupported(params, retina_size):
  """Compute the number of scale bands supported for a given retinal size.

  This ensures that at least one S2 unit can be computed for every scale band.

  :param params: Parameter settings for model.
  :param retina_size: Length of shorter edge of retina layer.
  :type retina_size: int
  :rtype: int
  :return: Number of scales.

  """
  min_retina_size = MinimumRetinaSize(params)
  num_scales = 0
  while min_retina_size < retina_size:
    num_scales += 1
    min_retina_size *= params.scale_factor
  return num_scales

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
  assert data.ndim == 3
  data -= data.mean(0)
  norms = np.sqrt((data**2).sum(0))
  norms[ norms < 1 ] = 1
  data /= norms
  return data

def GetS2ReceptiveField(params, scale, y, x, kwidth_offset=0):
  """Compute the receptive field for a given S2 unit.

  :param params: Model parameters.
  :type params: `glimpse.models.ml.Params`
  :param int scale: Unit's scale band.
  :param int y: Unit's vertical offset.
  :param int x: Unit's horizontal offset.
  :param int kwidth_offset: Offset for S2 kernel width (for models using
     multiple kernel sizes).
  :rtype: 4-tuple of int
  :returns: Bounding box (top, bottom, left, right) in image coordinates.

  """
  def map_min(params, i):
    i *= params.c1_sampling  # C1 to S1
    i *= params.s1_sampling  # S1 to (scaled) image
    return i
  def map_max(params, i):
    i = i * params.c1_sampling + params.c1_kwidth - 1  # C1 to S1
    i = i * params.s1_sampling + params.s1_kwidth - 1  # S1 to (scaled) image
    return i
  s2_kwidth = params.s2_kwidth[kwidth_offset]
  # Compute the receptive field in image coordinates for original image scale.
  unscaled_rf = [map_min(params, y),
      map_max(params, y + s2_kwidth),
      map_min(params, x),
      map_max(params, x + s2_kwidth) ]
  # Scale the receptive field.
  return tuple(int(i * params.scale_factor**scale) for i in unscaled_rf)

def PlotS2ReceptiveField(model, path, scale, y, x, s2_kwidth_offset=0):
  """Plot the location of an S2 unit's receptive field on a given image.

  :param model: Glimpse model for S2 unit.
  :param str path: Path to image on disk.
  :param int scale: Scale band of S2 unit.
  :param int y: Vertical offset of S2 unit.
  :param int x: Horizontal offset of S2 unit.
  :param int s2_kwidth_offset: Offset for kernel width of S2 unit.

  Requires matplotlib.

  """
  import matplotlib.pyplot as plt
  y0,y1,x0,x1 = GetS2ReceptiveField(model.params, scale, y, x, s2_kwidth_offset)
  st = BuildLayer(model, Layer.IMAGE, model.MakeState(path), save_all=False)
  plt.imshow(st[Layer.IMAGE])
  plt.yticks(()); plt.xticks(())
  plt.gca().add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0, facecolor='red',
      alpha=.5))
