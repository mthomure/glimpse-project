# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import numpy as np

from glimpse.backends import ACTIVATION_DTYPE
from glimpse.models.base.misc import SamplePatches
from glimpse.pools import SinglecorePool
from glimpse.util.callback import Callback
from glimpse.util.progress import MockProgressBar

def NormalizeLength(patches, epsilon=0):
  for patch in patches:
    patch /= np.linalg.norm(patch) + epsilon
  return patches

def NormalizeLocalContrast(patches, epsilon=None):
  """Normalize contrast per patch across variables *in-place*."""
  if epsilon is None:
    epsilon = 0
  for patch in patches:
    patch -= patch.mean()
    patch /= np.sqrt(patch.var() + epsilon)
  return patches

class GlobalContrastNormalizer():
  """Normalize contrast per variable across patches."""
  mean = None
  var = None
  epsilon = 0
  def fit(self, patches):
    """Estimate global contrast."""
    self.mean = patches.mean(0)
    self.std = np.sqrt(patches.var(0) + self.epsilon)
    return self
  def transform(self, patches):
    """Operates *in-place*."""
    patches -= self.mean
    patches /= self.std
    return patches
  def inverse_transform(self, patches):
    """Operates *in-place*."""
    assert self.mean is not None, "Must call fit() first."
    patches *= self.std
    patches += self.mean
    return patches

class Whitener():
  """Scale data to have equal variance in each eigen-subspace."""
  A = None  # forward transform
  B = None  # inverse transform
  epsilon = 0
  def __init__(self, epsilon=None):
    if epsilon is not None:
      self.epsilon = epsilon
  def fit(self, patches):
    """Estimate variance along eigen-dimensions."""
    C = np.cov(patches, rowvar=0)
    d, V = np.linalg.eigh(C)
    d = np.sqrt(d + self.epsilon)
    self.A = np.dot(np.dot(V.T, np.diag(1 / d)), V)
    # Get the inverse of A, using the fact that V is orthogonal and thus
    # V^T=V^-1.
    self.B = np.dot(np.dot(V.T, np.diag(d)), V)
    return self
  def transform(self, patches):
    return np.dot(self.A, patches.T).T
  def fit_transform(self, patches):
    return self.fit(patches).transform(patches)
  def inverse_transform(self, patches):
    return np.dot(self.B, patches.T).T

##### PATCH SAMPLING ###########################################################

def SamplePatchesFromImages(model, sample_layer, kernel_sizes, num_kernels,
    input_states, pool=None, normalize=False, progress=None):
  """Sample patches of multiple sizes from a set of images.

  :param model: Glimpse model.
  :type model: sub-class of :class:`glimpse.models.base.Model`
  :param sample_layer: Layer from which to sample.
  :type sample_layer: :class:`LayerSpec`
  :param kernel_sizes: Kernel widths to extract.
  :type kernel_sizes: int or list of int
  :param int num_kernels: Number of patches to sample for each kernel width.
  :param input_states: Initial network states from which to compute layer
     activity.
  :type input_states: list of `model.StateClass` or list of `str`
  :param pool: Worker pool used to evaluate the model.
  :param bool normalize: Whether to scale each kernel to have unit norm.
  :returns: Kernels, and their corresponding locations. Kernels are returned as
     a list of (N+1)-dimensional arrays, where N is the number of dimensions in
     a single kernel. The list axis and first axis of the array correspond to
     the kernel size and kernel offset, respectively.  Locations are returned as
     a 2D list of 4-tuples, where list axes correspond to kernel size and kernel
     offset, respectively. Each location is given as a 4-tuple, with elements
     corresponding to the input state index, scale, y-offset, and x-offset of
     the corresponding kernel.

  Note that the maximum kernel size must not be larger than the smallest
  response map at the given layer. To imprint a 7x7 kernel from C1, for example,
  the C1 response map for the lowest frequency band must be at least 7x7 in
  spatial extent.

  Examples:

  For each kernel size, sample 10 image patches from four (random) images.
  Kernels will be 7 and 11 pixel wide.

  >>> model = BaseModel()
  >>> images = np.random.random((4,100,100)).astype(ACTIVATION_DTYPE)
  >>> states = map(model.MakeState, images)
  >>> kernels,locations = SamplePatches(model, BaseLayer.IMAGE,
      kernel_sizes=(7,11), num_kernels=10, input_state=states)

  The result will contain a sub-list for each of the two kernel sizes.

  >>> assert len(kernels) == len(images)
  >>> assert all(len(ks) == 10 for ks in kernels)
  >>> assert len(locations) == len(images)
  >>> assert all(len(ls) == 10 for ls in locations)

  """
  if len(input_states) == 0:
    raise ValueError("No input states found")
  input_states = map(model.MakeState, input_states)
  if pool is None:
    pool = SinglecorePool()
  if progress is None:
    progress = MockProgressBar
  if not hasattr(kernel_sizes, '__len__'):
    kernel_sizes = (kernel_sizes,)
  if num_kernels == 0:
    return [ [[]] * len(kernel_sizes) ] * 2  # empty list of kernels and locs
  input_state_indices = np.arange(len(input_states))
  if num_kernels < len(input_states):
    # Take a random subset of images.
    np.random.shuffle(input_state_indices)
    input_state_indices = input_state_indices[:num_kernels]
    input_states = map(input_states.__getitem__, input_state_indices)
    num_imprinted_kernels = 1
  else:
    num_imprinted_kernels, extra = divmod(num_kernels, len(input_states))
    if extra > 0:
      num_imprinted_kernels += 1
  # The SamplePatches() function takes a mapping of patch size information,
  # where the key is a patch size and the value gives the number of patches for
  # that size (for each image). We create that mapping here.
  kernels_per_image = [ (size, num_imprinted_kernels) for size in kernel_sizes ]
  # Create a callback, which we can pass to the worker pool.
  sampler = Callback(_Sampler, model, sample_layer, kernels_per_image,
      normalize)
  # Compute layer activity, and sample patches.
  results_by_image = pool.map(sampler, input_states, progress = progress)
  # We now have a list of list of (patch array, location array) pairs, with one
  # sub-list state and one pair per kernel size.
  if len(results_by_image) != len(input_states):
    raise Exception("Internal error: expected results for %d images, got %d" %
        (len(input_states), len(results_by_image)))
  for results_by_ksize in results_by_image:
    if len(results_by_ksize) != len(kernel_sizes):
      raise Exception("Internal error: expected results for %d images, got %d"
          % (len(kernel_sizes), len(results_by_ksize)))
    for patches, locs in results_by_ksize:
      if len(patches) != num_imprinted_kernels:
        raise Exception("Internal error: expected %d patches per image, got %d"
            % (num_imprinted_kernels, len(patches)))
      if len(locs) != num_imprinted_kernels:
        raise Exception("Internal error: expected %d patches per image, got %d"
            % (num_imprinted_kernels, len(locs)))
  results_by_ksize = zip(*results_by_image)  # order by image, then kernel size
  patches = list()
  for results_for_img in results_by_ksize:
    patch_shape = results_for_img[0][0].shape  # patches element for first image
    patches_for_ksize = np.empty((len(results_for_img),) + patch_shape,
        dtype = ACTIVATION_DTYPE)
    for ksize, (ps, _) in enumerate(results_for_img):
      patches_for_ksize[ksize] = ps
    patches.append(patches_for_ksize)
  locs = np.array([ [ locs for _, locs in results_for_img ] for results_for_img
      in results_by_ksize ])
  # Concatenate each patch and location sub-list across images.
  patches = [ ps.reshape((-1,) + ps.shape[2:])
      for ps in patches ]  # concatenate across images (per ksize)
  locs = np.insert(locs, 0, 0,
      axis = -1)  # add a location element for image index
  # Annotate location tuples with image index.
  for idx in range(locs.shape[1]):
    locs[:, idx, :, 0] = input_state_indices[idx]
  locs = locs.reshape(locs.shape[0], -1,
      locs.shape[-1])  # concatenate across images (per ksize)
  # We may have sampled too many kernels (e.g., if the number of requested
  # patches is not an even multiple of the number of images). If so, crop
  # the results.
  if num_kernels < locs.shape[1]:
    indices = np.arange(locs.shape[1])
    np.random.shuffle(indices)
    indices = indices[:num_kernels]
    patches = [ ps[indices] for ps in patches ]
    locs = locs[:,indices]
  return patches, locs

def _Sampler(model, layer, counts, normalize, state):
  # :rtype: list of (patch array, location array) pairs, with one pair per
  #    kernel size
  patches = [ SamplePatches(model, layer, num_patches, patch_size, state)
      for patch_size, num_patches in counts ]
  if normalize:
    for patches_for_ksize in patches:
      for patch in patches_for_ksize[0]:
        norm = np.linalg.norm(patch)
        if norm == 0:
          logging.warn("Normalizing zero patch")
          patch[:] = 1.0 / sqrt(patch.size)
        else:
          patch /= norm
  return patches
