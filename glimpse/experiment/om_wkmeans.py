"""Weighted k-Means using weights based on foreground object overlap."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from collections import MutableMapping
import copy
import cPickle as pickle
import logging
import numpy as np
import os

from glimpse.util.learn import WeightedMiniBatchKMeans
from glimpse.models.ml.misc import GetS2ReceptiveField

class BaseMaskCache(dict):
  """An abstract base class for object mask caches."""

  def __init__(self, model, get_mask_path=None):
    self.model = model
    if get_mask_path is not None:
      self.get_mask_path = get_mask_path

  def __getitem__(self, img_path):
    """Get object mask for image.

    :param str img_path: Path to image for which object mask should be returned.
    :rtype: 2D array of float
    :return: Object mask as matrix of float, with values in [0,1).

    """
    # Note: this *does not* work as a module-level import, due to circular
    # import error.
    from glimpse.experiment import BuildLayer, Layer
    mask = dict.get(self, img_path)
    if mask is None:
      mask_path = self.get_mask_path(img_path)
      if mask_path is None or not os.path.exists(mask_path):
        raise IndexError
      st = BuildLayer(self.model, Layer.IMAGE, self.model.MakeState(mask_path))
      mask = st[Layer.IMAGE]
      self[img_path] = mask
    return mask

class MaskCache(BaseMaskCache):
  """A cache for object masks."""

  def __init__(self, model, mask_dir):
    super(MaskCache,self).__init__(model)
    if not os.path.isdir(mask_dir):
      raise ValueError("Mask directory not found: %s" % mask_dir)
    self.mask_dir = mask_dir

  def get_mask_path(self, img_path):
    """Get the path to an object mask on disk.

    :param str img_path: Path to image for which mask should be returned.
    :rtype: str
    :return: Path to object mask image.

    """
    if self.mask_dir is None:
      raise Exception("Must set mask directory")
    cls = os.path.basename(os.path.dirname(img_path))
    full_name = os.path.basename(img_path)
    name,ext = os.path.splitext(full_name)
    names = (full_name, name + '_mask' + ext, name + '_mask.png', name + '.png')
    for name in names:
      path = os.path.join(self.mask_dir, cls, name)
      if os.path.exists(path):
        return path
    return None

def PrototypeFGRatio(exp, masks):
  """Compute the foreground ratio for each prototype.

  The foreground ratio is the number of foreground pixels in the receptive
  field, divided by the total number of pixels in the receptive field. If the
  image has no foreground object, its foreground ratio is defined to be zero.

  :param masks: Mapping from image path to mask path. This will usually be a
     :class:MaskCache.
  :return: Foreground ratio for each imprinted prototype in experiment.
  :rtype: 1-d array of float

  """
  locs = exp.extractor.prototype_algorithm.locations
  weights = list()
  masks_found = 0
  for img_idx,scale,y,x in locs[0]:
    try:
      m = masks[exp.corpus.paths[img_idx]]
    except IndexError:
      weights.append(0)
      continue
    masks_found += 1
    y0,y1,x0,x1 = GetS2ReceptiveField(exp.extractor.model.params, scale, y, x)
    m = 1 - m[y0:y1,x0:x1]  # black means fg, white means bg
    w = m.sum() / m.size
    weights.append(w)
  logging.info("Found masks for %d (out of %d) patches", masks_found,
      len(locs[0]))
  return np.array(weights)

def LearnPatchesFromImages(exp, masks, num_samples, num_prototypes, pool,
    base_weight=None, progress=None):
  """Learn patch models by object-mask weighted k-Means clustering.

  Weights are given by the overlap between the image patch and the image's
  foreground object.

  :param masks: Mapping from image path to mask path. This will usually be a
     :class:MaskCache.
  :param int num_samples: Number of samples used to cluster via k-Means.
  :param int num_prototypes: Number of centroids used for k-Means.
  :param pool: Worker pool to use when extracting patches.
  :param float base_weight: Value added to all weights before learning.
  :param progress: Handler for incremental progress updates.
  :return: Learned patches.
  :rtype: array of float

  """
  # Note: the following *does not* work as a module-level import, due to a
  # circular import problem.
  from glimpse.experiment import MakePrototypes
  logging.info(("Learning %d prototypes per size by " % num_prototypes) +
      "(object-mask) weighted k-Means clustering.")
  assert len(exp.extractor.model.params.s2_kernel_widths) == 1, \
      "Multiple kernel sizes are not supported"
  if base_weight is None:
    base_weight = 0
  logging.info("\tnum_samples(%d), base_weight(%f)" % (num_samples, base_weight))
  MakePrototypes(exp, num_samples, 'imprint', pool, progress=progress)
  samples = exp.extractor.model.s2_kernels[0]
  weights = PrototypeFGRatio(exp, masks)
  weights += base_weight  # ensure background patches get some weight.
  kmeans = WeightedMiniBatchKMeans(n_clusters = num_prototypes).fit(
      samples.reshape(samples.shape[0], -1), weights)
  return kmeans.cluster_centers_.reshape(
      (kmeans.cluster_centers_.shape[0],) + samples.shape[1:])
