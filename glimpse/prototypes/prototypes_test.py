import numpy as np
import unittest

from glimpse.backends import ACTIVATION_DTYPE
from glimpse.models.ml import Model, Params, Layer
from glimpse.models.base import LayerSpec
from glimpse.models.base.layer import layer_builder
from glimpse.models.ml.param import SLayerOp
from glimpse.pools import SinglecorePool
from glimpse.util.garray import toimage
from glimpse.util.gtest import *

from .prototypes import *
from . import prototypes

class ReshapedLayer(Layer):

  RESHAPED_IMAGE = LayerSpec("r", "reshape-image", [Layer.IMAGE])

class ReshapedModel(Model):

  LayerClass = ReshapedLayer

  @layer_builder(ReshapedLayer.RESHAPED_IMAGE)
  def ReshapeImage(self, img):
    return img.reshape((1,) + img.shape)

class SampleAndLearnPatchesTests(unittest.TestCase):

  def test_default(self):
    num_patches = 11
    patch_widths = (7,9)
    samples_per_shape = [np.random.random((num_patches,1,w,w))
        for w in patch_widths]
    passed_samples_per_shape = list()
    def learner(num_patches_, samples, progress):
      self.assertEqual(num_patches, num_patches_)
      passed_samples_per_shape.append(samples)
      return samples
    sample_patches_from_images = RecordedFunctionCall((samples_per_shape,None))
    with MonkeyPatch(prototypes, 'SamplePatchesFromImages',
        sample_patches_from_images):
      patches_per_shape = SampleAndLearnPatches(None, [], learner, num_patches,
          patch_widths, ReshapedLayer.RESHAPED_IMAGE, None)
    self.assertTrue(sample_patches_from_images.called)
    # learner should have been passed elements of samples_per_shape
    self.assertEqual(len(passed_samples_per_shape), len(samples_per_shape))
    for ps,ss in zip(passed_samples_per_shape, samples_per_shape):
      self.assertTrue(np.allclose(ps, ss.reshape(num_patches,-1)))
    # patches should be the concatenation of learner results
    self.assertEqual(len(patches_per_shape), len(samples_per_shape))
    for ps,ss in zip(patches_per_shape, samples_per_shape):
      self.assertTrue(np.allclose(ps, ss))

  def test_errorOnBadPatchCount(self):
    num_patches = 11
    patch_widths = (7,9)
    samples_per_shape = [np.random.random((num_patches,1,w,w))
        for w in patch_widths]
    def learner(num_patches, samples, progress):
      return samples[:-2]
    sample_patches_from_images = RecordedFunctionCall((samples_per_shape,None))
    with MonkeyPatch(prototypes, 'SamplePatchesFromImages',
        sample_patches_from_images):
      with self.assertRaises(Exception):
        SampleAndLearnPatches(None, [], learner, num_patches, patch_widths,
            ReshapedLayer.RESHAPED_IMAGE, None)

  def test_errorOnBadPatchShape(self):
    num_patches = 11
    patch_widths = (7,9)
    samples_per_shape = [np.random.random((num_patches,1,w,w))
        for w in patch_widths]
    def learner(num_patches, samples, progress):
      return samples[:,:,:,:-2]
    sample_patches_from_images = RecordedFunctionCall((samples_per_shape,None))
    with MonkeyPatch(prototypes, 'SamplePatchesFromImages',
        sample_patches_from_images):
      with self.assertRaises(Exception):
        SampleAndLearnPatches(None, [], learner, num_patches, patch_widths,
            ReshapedLayer.RESHAPED_IMAGE, None)

  def test_noPatchesLearned(self):
    self.assertEqual(None, SampleAndLearnPatches(None, [], None, 0, None, None,
        pool=None))

NUM_SAMPLES = 10
NUM_PATCHES = 3
PATCH_SIZE = 5

class LearnerTests(unittest.TestCase):

  def testUniformRandom(self):
    num_patches = 11
    patch_shape = (13,15)
    patches = UniformRandom(num_patches, patch_shape, 0, 1)
    self.assertSequenceEqual(patches.shape, (num_patches,) + patch_shape)
    self.assertEqual(patches.dtype, ACTIVATION_DTYPE)

  def testHistogram(self):
    samples = np.random.random((NUM_SAMPLES,PATCH_SIZE))
    patches = Histogram(NUM_PATCHES, samples)
    self.assertSequenceEqual(patches.shape, (NUM_PATCHES,PATCH_SIZE))
    self.assertFalse(np.any(np.isnan(patches)))
    self.assertEqual(patches.dtype, ACTIVATION_DTYPE)

  def testNormalRandom(self):
    samples = np.random.random((NUM_SAMPLES,PATCH_SIZE))
    patches = NormalRandom(NUM_PATCHES, samples)
    self.assertSequenceEqual(patches.shape, (NUM_PATCHES,PATCH_SIZE))
    self.assertFalse(np.any(np.isnan(patches)))
    self.assertEqual(patches.dtype, ACTIVATION_DTYPE)

  def testKmeans(self):
    samples = np.random.random((NUM_SAMPLES,PATCH_SIZE))
    patches = Kmeans(NUM_PATCHES, samples)
    self.assertSequenceEqual(patches.shape, (NUM_PATCHES,PATCH_SIZE))
    self.assertFalse(np.any(np.isnan(patches)))
    self.assertEqual(patches.dtype, ACTIVATION_DTYPE)

  def testNearestKmeans(self):
    samples = np.random.random((NUM_SAMPLES,PATCH_SIZE))
    patches = NearestKmeans(NUM_PATCHES, samples)
    self.assertSequenceEqual(patches.shape, (NUM_PATCHES,PATCH_SIZE))
    self.assertFalse(np.any(np.isnan(patches)))
    self.assertEqual(patches.dtype, ACTIVATION_DTYPE)

  def _testKmedoids(self):  # disabled since Pycluster is unavailable
    samples = np.random.random((NUM_SAMPLES,PATCH_SIZE))
    patches = Kmedoids(NUM_PATCHES, samples)
    self.assertSequenceEqual(patches.shape, (NUM_PATCHES,PATCH_SIZE))
    self.assertFalse(np.any(np.isnan(patches)))
    self.assertEqual(patches.dtype, ACTIVATION_DTYPE)

  def testIca(self):
    samples = np.random.random((NUM_SAMPLES,PATCH_SIZE))
    patches = Ica(NUM_PATCHES, samples)
    self.assertSequenceEqual(patches.shape, (NUM_PATCHES,PATCH_SIZE))
    self.assertFalse(np.any(np.isnan(patches)))
    self.assertEqual(patches.dtype, ACTIVATION_DTYPE)

  def testPca(self):
    samples = np.random.random((NUM_SAMPLES,PATCH_SIZE))
    patches = Pca(NUM_PATCHES, samples)
    self.assertSequenceEqual(patches.shape, (NUM_PATCHES,PATCH_SIZE))
    self.assertFalse(np.any(np.isnan(patches)))
    self.assertEqual(patches.dtype, ACTIVATION_DTYPE)

  def testNmf(self):
    samples = np.random.random((NUM_SAMPLES,PATCH_SIZE))
    patches = Nmf(NUM_PATCHES, samples)
    self.assertSequenceEqual(patches.shape, (NUM_PATCHES,PATCH_SIZE))
    self.assertFalse(np.any(np.isnan(patches)))
    self.assertEqual(patches.dtype, ACTIVATION_DTYPE)

  def testSparsePca(self):
    samples = np.random.random((NUM_SAMPLES,PATCH_SIZE))
    patches = SparsePca(NUM_PATCHES, samples)
    self.assertSequenceEqual(patches.shape, (NUM_PATCHES,PATCH_SIZE))
    self.assertFalse(np.any(np.isnan(patches)))
    self.assertEqual(patches.dtype, ACTIVATION_DTYPE)

if __name__ == '__main__':
  unittest.main()
