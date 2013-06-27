"""Weighted k-Means using meta-feature quality model of Krupka et al."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import logging
import numpy as np
from scipy.stats.mstats import mquantiles
from sklearn import neighbors

from glimpse.util.learn import WeightedMiniBatchKMeans

def GetMetaFeatures(prototypes):
  """Compute "meta-features for a set of prototypes.

  :param prototypes: Prototype data.
  :return: Array of meta-features, with one row per prototype.
  :rtype: array of float

  """
  prototypes = prototypes.reshape(prototypes.shape[0], -1)
  qs = np.array([ mquantiles(p, prob=[.3,.7]) for p in prototypes ]).T
  meta_ftrs = np.array([ prototypes.mean(1), prototypes.std(1),
      prototypes.min(1), prototypes.max(1) ] + list(qs))
  return meta_ftrs.T

def TrainQualityModel(exp, num_regr_samples, pool, progress=None):
  """Train a patch quality model.

  :param int num_regr_samples: Number of patches to use for training the model.
  :param pool: Worker pool to use when extracting patches.
  :param progress: Handler for incremental progress updates.
  :return: A trained model.
  :rtype: sklearn.base.RegressorMixin

  """
  # Note: this *does not* work as a module-level import, due to circular import.
  from glimpse.experiment import (MakePrototypes, ComputeActivation,
      TrainAndTestClassifier, Layer)
  logging.info("Training prototype quality model based on meta-features")
  # make a local copy of the experiment
  MakePrototypes(exp, num_regr_samples, 'imprint', pool, progress=progress)
  ComputeActivation(exp, Layer.C2, pool, progress=progress)
  # estimate SVM parameters from feature vectors
  TrainAndTestClassifier(exp, Layer.C2)
  # Get ground-truth quality
  clf = exp.evaluation[0].results['classifier'].named_steps['learner']
  # Quality uses squared weights across support vectors, ignoring SV weights.
  quality = (clf.coef_**2).sum(0)
  meta_ftrs = _GetMetaFeatures(exp.extractor.model.s2_kernels[0])
  # estimate regression parameters from SVM weights
  logging.info("Estimate parameters of regression model")
  quality_model = neighbors.KNeighborsRegressor(n_neighbors=4).fit(meta_ftrs,
      quality)
  return quality_model

def LearnPatchesFromImages(exp, num_regr_samples, num_samples, num_prototypes,
    pool, progress=None):
  """Learn patch models by meta-feature weighted k-Means clustering.

  Weights are given by a feature quality prediction model using prototype
  "meta-features".

  :param int num_regr_samples: Number of patches used to train quality
     prediction regression model.
  :param int num_samples: Number of samples used to cluster via k-Means.
  :param int num_prototypes: Number of centroids used for k-Means.
  :param pool: Worker pool to use when extracting patches.
  :param progress: Handler for incremental progress updates.
  :return: Learned patches.
  :rtype: array of float

  """
  # Note: the following *does not* work as a module-level import, due to a
  # circular import problem.
  from glimpse.experiment import MakePrototypes
  logging.info("Learning %d prototypes per size by (meta-feature) weighted "
      "k-Means clustering.", num_prototypes)
  assert len(exp.extractor.model.params.s2_kernel_widths) == 1, \
      "Multiple kernel sizes are not supported"
  logging.info("\tnum_regr_samples(%d), num_samples(%d)", num_regr_samples,
      num_samples)
  quality_model = _TrainQualityModel(exp, num_regr_samples, pool,
      progress=progress)
  # sample C1 patches
  MakePrototypes(exp, num_samples, 'imprint', pool, progress=progress)
  samples = exp.extractor.model.s2_kernels[0]
  # estimate patch weights using the quality model
  weights = quality_model.predict(_GetMetaFeatures(samples))
  # choose prototypes by weighted k-means
  logging.info("Estimate C1 Clusters using weighted k-Means")
  kmeans = WeightedMiniBatchKMeans(n_clusters = num_prototypes).fit(
      samples.reshape(samples.shape[0], -1), weights)
  prototypes = kmeans.cluster_centers_
  return prototypes.reshape((prototypes.shape[0],) + samples.shape[1:])
