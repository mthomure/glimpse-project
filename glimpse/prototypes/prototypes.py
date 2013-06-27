"""Functions for learning S2 prototypes."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import logging
import numpy as np
import operator

from glimpse.backends import ACTIVATION_DTYPE
from glimpse.util.grandom import HistogramSampler
from glimpse.util.kernel import MakeUniformRandomKernels
from .utils import SamplePatchesFromImages

def UniformRandom(num_patches, patch_shape, low, high):
  """Create patches by sampling components uniformly."""
  logging.info("Sampling %d patches from multivariate " % num_patches +
      "uniform distribution over [%s,%s)" % (low, high))
  if low >= high:
    raise ValueError("Low value (%s) must be less than high value (%s)" % (low,
        high))
  return MakeUniformRandomKernels(num_patches, patch_shape, normalize=False,
      low=low, high=high).astype(ACTIVATION_DTYPE)

def SampleAndLearnPatches(model, images, learner, num_patches,
    patch_widths, layer, pool, num_samples=None, progress=None):
  """Draw samples of layer activity and build a simple patch model.

  :param model: Glimpse model with which to generate layer activity.
  :param layer: The layer from which to sample patches.
  :param images: list of model states (e.g., image paths).
  :param callable learner: Patch model estimator, with the signature
     `learner(num_patches, samples, progress=progress)`
     where `samples` is an (N+1)-dimensional array of samples.
  :param int num_patches: Number of patches to generate.
  :param patch_widths: Size of patches to draw.
  :type patch_widths: list of int
  :param pool: Worker pool.
  :param int num_samples: Number of patch samples to draw. By default, the
     number of samples is 10x the number of desired patches.
  :param progress: Callback for progress updates.
  :rtype: list of (N+1)-dimensional array, where N is the number of dimensions
     in a patch.
  :returns: Set of patches for each patch size.

  For each size in `patch_widths`, a set of `num_samples` is drawn from the
  images and modelled via the `learner`.

  """
  if num_patches == 0:
    return None
  if num_samples == 0 or num_samples is None:
    # Allow 10 patches per cluster
    num_samples = num_patches * 10
  logging.info("Sampling %d layer %s patches at %d sizes from %d images" % (
      num_samples, layer, len(patch_widths), len(images)))
  samples_per_shape,_ = SamplePatchesFromImages(model, layer, patch_widths,
      num_samples, images, pool=pool, normalize=False, progress=progress)
  patches_per_shape = list()
  for idx,samples in enumerate(samples_per_shape):
    samples = samples.astype(ACTIVATION_DTYPE)  # override caller-passed type
    sample_shape = samples.shape[1:]
    samples = samples.reshape(samples.shape[0], -1)
    patches = learner(num_patches, samples, progress=progress)
    if len(patches) != num_patches:
      raise Exception("Patch learner returned wrong number of patches: "
          "expected %d but got %d" % (num_patches, len(patches)))
    if patches.shape[1] != samples.shape[1]:
      raise Exception("Patch learner returned patches with wrong shape: "
          "expected %s but got %s" % (sample_shape, patches.shape[1:]))
    patches = patches.reshape((num_patches,) + sample_shape)
    patches = patches.astype(ACTIVATION_DTYPE)  # override learner-result type
    patches_per_shape.append(patches)
  return patches_per_shape

def Histogram(num_patches, samples, progress=None):
  """Create patches from a histogram over sample values.

  :param int num_patches: number of patches to create
  :type samples: 2D array
  :param samples: example patches
  :param progress: ignored
  :rtype: 2D array with `num_patches` rows and N columns, where N is the number
     of columns in `samples`.
  :return: created patches

  """
  logging.info(("Learning %d prototypes per size " % num_patches) +
      "by imprinting + histogram sampling")
  samples = samples.astype(ACTIVATION_DTYPE)
  hist = HistogramSampler(samples.flat)
  return hist.Sample((num_patches, samples.shape[1]),
      dtype=ACTIVATION_DTYPE)

def NormalRandom(num_patches, samples, progress=None):
  """Estimate parameters of a normal distribution and sample from it.

  :param int num_patches: number of patches to create
  :type samples: 2D array
  :param samples: example patches
  :param progress: ignored
  :rtype: 2D array with `num_patches` rows and N columns, where N is the number
     of columns in `samples`.
  :return: created patches

  """
  logging.info("Learning %d prototypes by imprinting + Gaussian sampling" %
      num_patches)
  samples = samples.astype(ACTIVATION_DTYPE)
  return np.random.normal(samples.mean(), samples.std(), (num_patches,) +
      samples.shape[1:]).astype(ACTIVATION_DTYPE)

def _KmeansFit(num_patches, samples, batch):
  import sklearn.cluster
  if batch:
    logging.info("  using batch mode")
    kmeans = sklearn.cluster.KMeans(n_clusters=num_patches, n_jobs=-1)
    kmeans.fit(samples)
  else:
    logging.info("  using sequential mode")
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_patches)
    # TODO: break `samples` into smaller batches and update `progress`.
    kmeans.fit(samples)
  return kmeans

def Kmeans(num_patches, samples, progress=None, batch=False):
  """Estimate patches as centroids of samples using k-Means.

  :param int num_patches: number of patches to create
  :type samples: 2D array
  :param samples: example patches
  :param progress: ignored
  :param bool batch: whether to learn using batch or online k-Means
  :rtype: 2D array with `num_patches` rows and N columns, where N is the number
     of columns in `samples`.
  :return: created patches

  """
  logging.info("Learning %d prototypes per size by k-Means clustering" %
      num_patches)
  kmeans = _KmeansFit(num_patches, samples, batch)
  return kmeans.cluster_centers_.astype(ACTIVATION_DTYPE)

def NearestKmeans(num_patches, samples, progress=None, batch=False):
  """Estimate patches as centroids of samples using nearest-value k-Means.

  This performs k-Means cluster of the samples, and then replaces each centroid
  with its nearest cluster element.

  :param int num_patches: number of patches to create
  :type samples: 2D array
  :param samples: example patches
  :param progress: ignored
  :param bool batch: whether to learn using batch or online k-Means
  :rtype: 2D array with `num_patches` rows and N columns, where N is the number
     of columns in `samples`.
  :return: created patches

  """
  logging.info(("Learning %d prototypes " % num_patches) +
      "per size by nearest-k-Means clustering")
  kmeans = _KmeansFit(num_patches, samples, batch)
  cluster_distance = kmeans.transform(samples)
  cluster_ids = cluster_distance.argmin(1)  # find nearest centroid
  patches = np.empty((num_patches,) + samples.shape[1:], samples.dtype)
  for k in range(num_patches):
    cluster = (cluster_ids == k)
    ds = cluster_distance[cluster][:,k]  # distance to instances in cluster
    best_instance_idx = ds.argmin()
    # find patch in cluster with minimum distance to center
    patches[k] = samples[cluster][ds.argmin()]
  return patches.astype(ACTIVATION_DTYPE)

def Kmedoids(num_patches, samples, progress=None):
  """Estimate patches as centroids of samples using k-Medoids.

  This requires the `Pycluster` library to be installed.

  :param int num_patches: number of patches to create
  :type samples: 2D array
  :param samples: example patches
  :param progress: ignored
  :rtype: 2D array with `num_patches` rows and N columns, where N is the number
     of columns in `samples`.
  :return: created patches

  """
  logging.info("Learning %d prototypes per size by k-Medoids clustering" %
      num_patches)
  import Pycluster
  dist = Pycluster.distancematrix(samples)
  cluster_ids, _, _ = Pycluster.kmedoids(dist, nclusters=num_patches)
  # `cluster_ids` contains `num_patches` unique values, each of which is
  # the index of the medoid for a different cluster.
  return samples[np.unique(cluster_ids)].astype(ACTIVATION_DTYPE)

def Ica(num_patches, samples, progress=None):
  """Estimate patches using Independent Components Analysis.

  :param int num_patches: number of patches to create
  :type samples: 2D array
  :param samples: example patches
  :param progress: ignored
  :rtype: 2D array with `num_patches` rows and N columns, where N is the number
     of columns in `samples`.
  :return: created patches

  """
  logging.info("Learning %d prototypes per size by ICA" %
      num_patches)
  import sklearn.decomposition
  ica = sklearn.decomposition.FastICA(n_components=num_patches, whiten=True)
  ica.fit(samples)
  # If feature detectors are implemented in terms of dot-products, use the
  # filters as given by the unmixing matrix. If feature detectors are
  # implemented in terms of L2 distances, use the bases as given by the mixing
  # matrix.
  return ica.get_mixing_matrix().T.astype(ACTIVATION_DTYPE)

def Pca(num_patches, samples, progress=None):
  """Estimate patches using Principal Components Analysis.

  :param int num_patches: number of patches to create
  :type samples: 2D array
  :param samples: example patches
  :param progress: ignored
  :rtype: 2D array with `num_patches` rows and N columns, where N is the number
     of columns in `samples`.
  :return: created patches

  """
  logging.info("Learning %d prototypes per size by PCA" % num_patches)
  import sklearn.decomposition
  pca = sklearn.decomposition.RandomizedPCA(n_components=num_patches,
      whiten=True)
  pca.fit(samples)
  return (pca.components_ + pca.mean_).astype(ACTIVATION_DTYPE)

def SparsePca(num_patches, samples, progress=None):
  """Estimate patches using Sparse Principal Components Analysis.

  See `sklearn.decomposition.MiniBatchSparsePCA` for more information.

  :param int num_patches: number of patches to create
  :type samples: 2D array
  :param samples: example patches
  :param progress: ignored
  :rtype: 2D array with `num_patches` rows and N columns, where N is the number
     of columns in `samples`.
  :return: created patches

  """
  logging.info("Learning %d prototypes per size by SparsePCA" %
      num_patches)
  import sklearn.decomposition
  pca = sklearn.decomposition.MiniBatchSparsePCA(n_components=num_patches,
      alpha=0.8, n_iter=100, batch_size=3)
  return pca.fit(samples).components_.astype(ACTIVATION_DTYPE)

def Nmf(num_patches, samples, progress=None):
  """Estimate patches using Non-negative Matrix Factorization.

  :param int num_patches: number of patches to create
  :type samples: 2D array
  :param samples: example patches
  :param progress: ignored
  :rtype: 2D array with `num_patches` rows and N columns, where N is the number
     of columns in `samples`.
  :return: created patches

  """
  logging.info("Learning %d prototypes per size by NMF" %
      num_patches)
  import sklearn.decomposition
  nmf = sklearn.decomposition.NMF(n_components=num_patches, init='nndsvda',
      beta=5.0, tol=5e-3, sparseness='components')
  return nmf.fit(samples).components_.astype(ACTIVATION_DTYPE)
