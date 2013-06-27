"""Algorithms for generating prototypes.

A prototype algorithm is a callable with the signature:
   prototypes = algorithm(num_prototypes, model, make_training_exp, pool, progress=None)
where `images` is a generator of image paths.

See also :func:`glimpse.experiment.MakePrototypes`.

"""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import logging
import os

from glimpse.prototypes import *
from glimpse import prototypes
from . import mf_wkmeans, om_wkmeans

class ImprintAlg(object):
  """Learn prototypes by imprinting from training images."""

  #: Image locations from which samples were drawn.
  locations = None

  def __init__(self, record_locations=True):
    self.record_locations = record_locations
    self.locations = None

  def __call__(self, num_prototypes, model, make_training_exp, pool, progress):
    patch_widths = model.params.s2_kernel_widths
    images = make_training_exp().corpus.paths
    logging.info("Learning %d prototypes at %d sizes from %d images by "
        "imprinting", num_prototypes, len(patch_widths), len(images))
    patches_per_shape,sample_locations = SamplePatchesFromImages(model,
        model.LayerClass.C1, patch_widths, num_prototypes, images, pool=pool,
        normalize=False, progress=progress)
    if model.params.s2_kernels_are_normed:
      patches_per_shape = map(prototypes.NormalizeLength, patches_per_shape)
    if self.record_locations:
      self.locations = sample_locations
    return patches_per_shape

# deprecated name
ImprintProtoAlg = ImprintAlg

class ShuffledAlg(ImprintAlg):
  """Learn prototypes by shuffling a set of imprinted prototypes."""

  def __call__(self, num_prototypes, model, make_training_exp, pool, progress):
    patches_per_shape = super(ShuffledAlg, self).__call__(num_prototypes, model,
        make_training_exp, pool, progress)
    logging.info("Shuffling learned prototypes")
    for ps in patches_per_shape:
      for p in ps:
        np.random.shuffle(p.flat)
    return patches_per_shape

class UniformAlg():
  """Create prototypes by sampling components uniformly."""

  def __init__(self, low=0, high=1):
    #: Lower limit of uniform distribution.
    self.low = low
    #: Upper limit of uniform distribution.
    self.high = high

  def __call__(self, num_prototypes, model, make_training_exp, pool, progress):
    patch_shapes = model.params.s2_kernel_shapes
    logging.info("Sampling %d prototypes at %d sizes " % (num_prototypes,
        len(patch_shapes)) + "from uniform random distribution")
    # XXX `progress` is ignored
    patches_per_shape = [ prototypes.UniformRandom(num_prototypes, kshape,
        self.low, self.high) for kshape in patch_shapes ]
    if model.params.s2_kernels_are_normed:
      patches_per_shape = map(prototypes.NormalizeLength, patches_per_shape)
    return patches_per_shape

# deprecated
UniformProtoAlg = UniformAlg

class _SimpleLearningAlg():

  #: Number of samples from which to learn.
  num_samples = None

  def __init__(self, learn_patches=None):
    # We conditionally set this attribute, since sub-classes directly implement
    # learn_patches().
    if learn_patches is not None:
      self.learn_patches = learn_patches

  def __call__(self, num_prototypes, model, make_training_exp, pool, progress):
    patch_shapes = model.params.s2_kernel_shapes
    exp = make_training_exp()
    paths = exp.corpus.paths
    logging.info("Learning %d prototypes at %d sizes " % (num_prototypes,
        len(patch_shapes)) + "from %d images" % len(paths))
    patches_per_shape = prototypes.SampleAndLearnPatches(model, paths,
        self.learn_patches, num_prototypes, model.params.s2_kernel_widths,
        model.LayerClass.C1, pool, num_samples=self.num_samples,
        progress=progress)
    if model.params.s2_kernels_are_normed:
      patches_per_shape = map(prototypes.NormalizeLength, patches_per_shape)
    return patches_per_shape

# deprecated name
LearningAlg = _SimpleLearningAlg

#: Learn prototypes from a histogram over sample C1 values.
HistogramAlg = lambda: _SimpleLearningAlg(prototypes.Histogram)

#: Create prototypes by sampling elements from a standard normal distribution.
NormalAlg = lambda: _SimpleLearningAlg(prototypes.NormalRandom)

#: Learn prototypes as centroids of samples using nearest-value k-Means.
NearestKmeansAlg = lambda: _SimpleLearningAlg(prototypes.NearestKmeans)

#: Learn prototypes as centroids of samples using k-Medoids.
KmedoidsAlg = lambda: _SimpleLearningAlg(prototypes.Kmedoids)

#: Learn prototypes using Independent Components Analysis.
IcaAlg = lambda: _SimpleLearningAlg(prototypes.Ica)

#: Learn prototypes using Principal Components Analysis.
PcaAlg = lambda: _SimpleLearningAlg(prototypes.Pca)

#: Learn prototypes using Sparse Principal Components Analysis.
SparsePcaAlg = lambda: _SimpleLearningAlg(prototypes.SparsePca)

#: Learn prototypes using Non-negative Matrix Factorization.
NmfAlg = lambda: _SimpleLearningAlg(prototypes.Nmf)

class KmeansAlg(_SimpleLearningAlg):
  """Learn prototypes as centroids of C1 patches using k-Means."""

  normalize_contrast = False
  whiten = False
  unwhiten = False
  batch = False

  def learn_patches(self, num_patches, samples, progress=None):
    """Learn patches by k-Means. Modifies `samples` array."""
    whitener = None
    if self.normalize_contrast:
      samples = prototypes.NormalizeLocalContrast(samples)
    if self.whiten:
      whitener = prototypes.Whitener().fit(samples)
      samples = whitener.transform(samples)
    patches = prototypes.Kmeans(num_patches, samples, progress, self.batch)
    if self.unwhiten and whitener is not None:
      patches = whitener.inverse_transform(patches)
    return patches

# deprecated
KmeansProtoAlg = KmeansAlg

class _WeightedKmeansAlg():

  #: Number of weighted samples to use for learning.
  num_samples = None

  def __call__(self, num_prototypes, model, make_training_exp, pool, progress):
    num_samples = self.num_samples
    if num_samples == 0 or num_samples is None:
      # Allow 10 patches per cluster
      num_samples = num_prototypes * 10
    exp = make_training_exp()
    if len(exp.extractor.model.params.s2_kernel_widths) > 1:
      raise ValueError("Only single-sized S2 prototypes are supported")
    logging.info(("Learning %d prototypes at 1 size " % num_prototypes)
         + "from %d images" % len(exp.corpus.paths))
    patches = self.LearnPatches(exp, num_samples, num_prototypes, pool,
        progress)
    if model.params.s2_kernels_are_normed:
      patches = prototypes.NormalizeLength(patches)
    return (patches,)

class MFWKmeansAlg(_WeightedKmeansAlg):
  """Learn patch models by meta-feature weighted k-Means clustering.

  .. seealso::
     :func:`mf_wkmeans.LearnPatchesFromImages
     <glimpse.experiment.mf_wkmeans.LearnPatchesFromImages>`.

  """

  #: Number of samples with which to train regr model
  num_regr_samples = None

  def LearnPatches(self, exp, num_samples, num_prototypes, pool, progress):
    num_regr_samples = self.num_regr_samples
    if num_regr_samples == 0 or num_regr_samples is None:
      num_regr_samples = 250  # this was found to be effective on Caltech101
    return mf_wkmeans.LearnPatchesFromImages(exp, num_regr_samples, num_samples,
        num_prototypes, pool, progress=progress)

class OMWKmeansAlg(_WeightedKmeansAlg):
  """Learn patch models by object-mask weighted k-Means clustering.

  .. seealso::
     :func:`om_wkmeans.LearnPatchesFromImages
     <glimpse.experiment.om_wkmeans.LearnPatchesFromImages>`.

  """

  #: Directory containing object masks.
  mask_dir = None
  #: Weight added for all patches.
  base_weight = None

  def LearnPatches(self, exp, num_samples, num_prototypes, pool, progress):
    masks = om_wkmeans.MaskCache(exp.extractor.model, self.mask_dir)
    return om_wkmeans.LearnPatchesFromImages(exp, masks, num_samples,
        num_prototypes, pool, base_weight=self.base_weight, progress=progress)

_ALGORITHMS = dict(
    imprint = ImprintAlg,
    uniform = UniformAlg,
    shuffle = ShuffledAlg,
    histogram = HistogramAlg,
    normal = NormalAlg,
    kmeans = KmeansAlg,
    nearest_kmeans = NearestKmeansAlg,
    kmedoids = KmedoidsAlg,
    pca = PcaAlg,
    ica = IcaAlg,
    nmf = NmfAlg,
    sparse_pca = SparsePcaAlg,
    meta_feature_wkmeans = MFWKmeansAlg,
    object_mask_wkmeans = OMWKmeansAlg,
    )

def GetAlgorithmNames():
  """Lookup the name of all available prototype algorithm.

  :rtype: list of str
  :returns: Name of all known prototype algorithms, any of which can be passed
     to :func:`ResolveAlgorithm`.

  """
  return _ALGORITHMS.keys()

def ResolveAlgorithm(alg):
  """Lookup a prototype algorithm by name.

  :param str alg: Name of a prototype algorithm, as defined by :func:`GetNames`.
     This value is not case sensitive.
  :rtype: callable
  :returns: Prototype algorithm.

  """
  try:
    return _ALGORITHMS[alg.lower()]
  except KeyError:
    raise ExpError("Unknown prototype algorithm: %s" % alg)
