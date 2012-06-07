# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# Unit tests for the glab module.

import os
from os.path import join as pjoin
import random
import unittest

from glimpse import glab
from glimpse import pools
from glimpse.util import TempDir, TouchFile

def MakeDirs(dir, *args, **kwargs):
  """Create a directory hierarchy given by a dictionary of lists of strings."""
  for k in args:
    TouchFile(pjoin(dir, k))
  for k, v in kwargs.items():
    path = pjoin(dir, k)
    os.mkdir(path)
    if isinstance(v, dict):
      MakeDirs(path, **v)
    else:
      MakeDirs(path, *v)

def MockDirs(*args, **kwargs):
  temp_dir = TempDir()
  MakeDirs(temp_dir.dir, *args, **kwargs)
  return temp_dir

EXAMPLE_CORPUS = glab.GetExampleCorpus()
EXAMPLE_LARGE_CORPUS = glab.GetLargeExampleCorpus()

def ImageDict(num_classes, start = 1, end = 5):
  return dict(('cls%d' % c, [ 'cls%d_img%d.jpg' % (c, i)
      for i in range(start, end) ])
      for c in range(1, num_classes + 1))

CLASSES2 = ('cls1', 'cls2')
TRAIN_IMAGES2 = ImageDict(2, end = 3)
TEST_IMAGES2 = ImageDict(2, start = 3, end = 5)
IMAGES2 = ImageDict(2, end = 5)

CLASSES3 = ('cls1', 'cls2', 'cls3')
TRAIN_IMAGES3 = ImageDict(3, end = 3)
TEST_IMAGES3 = ImageDict(3, start = 3, end = 5)
IMAGES3 = ImageDict(3, end = 5)

NUM_PROTOTYPES = 10

class TestGlab(unittest.TestCase):

  def setUp(self):
    glab.Reset()

  def testSetCorpus_binaryNoClasses(self):
    self._testSetCorpus(IMAGES2, expected_classes = None)

  def testSetCorpus_binaryWithClasses(self):
    self._testSetCorpus(IMAGES2, expected_classes = CLASSES2)

  def testSetCorpus_multiclassNoClasses(self):
    self._testSetCorpus(IMAGES3, expected_classes = None)

  def testSetCorpus_multiclassWithClasses(self):
    expected_classes = list(CLASSES3)
    # Test different class orderings.
    for _ in range(3):
      random.shuffle(expected_classes)
      self._testSetCorpus(IMAGES3, expected_classes = expected_classes)

  def testSetCorpusSubdirs_binary_noClasses(self):
    self._testSetCorpusSubdirs(IMAGES2, None)

  def testSetCorpusSubdirs_binary_withClasses(self):
    self._testSetCorpusSubdirs(IMAGES2, CLASSES2)

  def testSetCorpusSubdirs_multiclass_noClasses(self):
    self._testSetCorpusSubdirs(IMAGES3, None)

  def testSetCorpusSubdirs_multiclass_withClasses(self):
    expected_classes = list(CLASSES3)
    # Test different class orderings.
    for _ in range(3):
      random.shuffle(expected_classes)
      self._testSetCorpusSubdirs(IMAGES3, expected_classes)

  def testSetTrainTestSplitFromDirs_binary_noClasses(self):
    self._testSetTrainTestSplitFromDirs(TRAIN_IMAGES2, TEST_IMAGES2, None)

  def testSetTrainTestSplitFromDirs_binary_withClasses(self):
    self._testSetTrainTestSplitFromDirs(TRAIN_IMAGES2, TEST_IMAGES2, CLASSES2)

  def testSetTrainTestSplitFromDirs_multiclass_noClasses(self):
    self._testSetTrainTestSplitFromDirs(TRAIN_IMAGES3, TEST_IMAGES3, None)

  def testSetTrainTestSplitFromDirs_multiclass_withClasses(self):
    expected_classes = list(CLASSES3)
    # Test different class orderings.
    for _ in range(3):
      random.shuffle(expected_classes)
      self._testSetTrainTestSplitFromDirs(TRAIN_IMAGES3, TEST_IMAGES3,
          expected_classes)

  def _testSetCorpus(self, images, expected_classes):
    root = MockDirs(**images)
    glab.SetCorpus(root.dir, classes = expected_classes)
    e = glab.GetExperiment()
    self.assertEqual(e.corpus, root.dir)
    self.assertNotEqual(e.classes, None)
    if expected_classes == None:
      self.assertEqual(sorted(e.classes), sorted(images.keys()))
    else:
      self.assertEqual(e.classes, expected_classes)
    for idx in range(len(e.classes)):
      actual_images = sorted(map(os.path.basename,
          e.train_images[idx] + e.test_images[idx]))
      expected_images = sorted(images[e.classes[idx]])
      self.assertEqual(actual_images, expected_images)

  def _testSetCorpusSubdirs(self, images, expected_classes):
    root = MockDirs(**images)
    if expected_classes == None:
      subdirs = [ pjoin(root.dir, cls) for cls in images.keys() ]
    else:
      subdirs = [ pjoin(root.dir, cls) for cls in expected_classes ]
    glab.SetCorpusSubdirs(subdirs, classes = expected_classes, balance = False)
    e = glab.GetExperiment()
    self.assertNotEqual(e.classes, None)
    if expected_classes == None:
      self.assertEqual(sorted(e.classes), sorted(images.keys()))
    else:
      self.assertEqual(e.classes, expected_classes)
    for idx in range(len(e.classes)):
      actual_images = sorted(map(os.path.basename,
          e.train_images[idx] + e.test_images[idx]))
      expected_images = sorted(images[e.classes[idx]])
      self.assertEqual(actual_images, expected_images)

  def _testSetTrainTestSplitFromDirs(self, train_images, test_images,
      expected_classes):
    root = MockDirs(train = train_images, test = test_images)
    glab.SetTrainTestSplitFromDirs(pjoin(root.dir, 'train'),
        pjoin(root.dir, 'test'), classes = expected_classes)
    e = glab.GetExperiment()
    self.assertNotEqual(e.classes, None)
    if expected_classes == None:
      self.assertEqual(sorted(e.classes), sorted(train_images.keys()))
    else:
      self.assertEqual(e.classes, expected_classes)
    for idx in range(len(e.classes)):
      cls = e.classes[idx]
      actual_images = sorted(map(os.path.basename, e.train_images[idx]))
      self.assertEqual(actual_images, sorted(train_images[cls]))
      actual_images = sorted(map(os.path.basename, e.test_images[idx]))
      self.assertEqual(actual_images, sorted(test_images[cls]))

  def testStoreExperiment_empty(self):
    temp_dir = TempDir()
    glab.StoreExperiment(pjoin(temp_dir.dir, 'dat'))

  def testStoreExperiment_notEmpty(self):
    temp_dir = TempDir()
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.ImprintS2Prototypes(10)
    glab.RunSvm()
    old_exp = glab.GetExperiment()
    exp_path = pjoin(temp_dir.dir, 'dat')
    glab.StoreExperiment(exp_path)
    new_exp = glab.LoadExperiment(exp_path)
    self.assertEqual(new_exp, old_exp)

class _TestGlabForModel(unittest.TestCase):

  MODEL_CLASS = None
  MODEL_LAYER = "c1"

  def setUp(self):
    glab.Reset()
    glab.SetModelClass(self.MODEL_CLASS)
    glab.SetLayer(self.MODEL_LAYER)
    #~ glab.SetPool(pools.SinglecorePool())

  def testRunSvm(self):
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.RunSvm()
    e = glab.GetExperiment()
    self.assertNotEqual(e.train_results['accuracy'], None)
    self.assertNotEqual(e.test_results['accuracy'], None)

  def testCrossValidateSvm(self):
    glab.SetCorpus(EXAMPLE_LARGE_CORPUS)
    glab.CrossValidateSvm()
    e = glab.GetExperiment()
    self.assertEqual(e.train_results, None)
    self.assertNotEqual(e.test_results['accuracy'], None)

  def testGetImageFeatures(self):
    images = glab.GetExampleImages()[:2]
    glab.SetLayer("c1")
    features = glab.GetImageFeatures(images)
    self.assertEqual(len(features), 2)
    self.assertEqual(len(features[0]), len(features[1]))

class _TestGlabForModelC2(_TestGlabForModel):

  MODEL_CLASS = None
  MODEL_LAYER = "c2"

  def setUp(self):
    glab.Reset()
    glab.SetModelClass(self.MODEL_CLASS)
    glab.SetLayer(self.MODEL_LAYER)
    #~ glab.SetPool(pools.SinglecorePool())

  def testRunSvm(self):
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.ImprintS2Prototypes(NUM_PROTOTYPES)
    glab.RunSvm()
    e = glab.GetExperiment()
    self.assertNotEqual(e.train_results['accuracy'], None)
    self.assertNotEqual(e.test_results['accuracy'], None)

  def testCrossValidateSvm(self):
    glab.SetCorpus(EXAMPLE_LARGE_CORPUS)
    glab.ImprintS2Prototypes(NUM_PROTOTYPES)
    glab.CrossValidateSvm()
    e = glab.GetExperiment()
    self.assertEqual(e.train_results, None)
    self.assertNotEqual(e.test_results['accuracy'], None)

  def testImprintS2Prototypes(self):
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.ImprintS2Prototypes(NUM_PROTOTYPES)
    e = glab.GetExperiment()
    # S2 kernels are stored as a list of arrays, with one list entry per kernel
    # size. Check that we've imprinted NUM_PROTOTYPES patches for each kernel
    # size.
    self.assertEqual(map(len, e.model.s2_kernels),
        [ NUM_PROTOTYPES ] * len(e.model.s2_kernels) )

  def testMakeUniformRandomS2Prototypes(self):
    glab.MakeUniformRandomS2Prototypes(NUM_PROTOTYPES)
    e = glab.GetExperiment()
    self.assertEqual(map(len, e.model.s2_kernels),
        [ NUM_PROTOTYPES ] * len(e.model.s2_kernels) )

  def testMakeShuffledRandomS2Prototypes(self):
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.MakeShuffledRandomS2Prototypes(NUM_PROTOTYPES)
    e = glab.GetExperiment()
    self.assertEqual(map(len, e.model.s2_kernels),
        [ NUM_PROTOTYPES ] * len(e.model.s2_kernels) )

  def testMakeHistogramRandomS2Prototypes(self):
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.MakeHistogramRandomS2Prototypes(NUM_PROTOTYPES)
    e = glab.GetExperiment()
    self.assertEqual(map(len, e.model.s2_kernels),
        [ NUM_PROTOTYPES ] * len(e.model.s2_kernels) )

  def testMakeNormalRandomS2Prototypes(self):
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.MakeNormalRandomS2Prototypes(NUM_PROTOTYPES)
    e = glab.GetExperiment()
    self.assertEqual(map(len, e.model.s2_kernels),
        [ NUM_PROTOTYPES ] * len(e.model.s2_kernels) )

class TestGlabForHmax(_TestGlabForModelC2):

  MODEL_CLASS = "hmax"

class TestGlabForMl(_TestGlabForModelC2):

  MODEL_CLASS = "ml"

class TestGlabForViz2(_TestGlabForModelC2):

  MODEL_CLASS = "viz2"

class TestGlabForHmaxC1(_TestGlabForModel):

  MODEL_CLASS = "hmax"

class TestGlabForMlC1(_TestGlabForModel):

  MODEL_CLASS = "ml"

class TestGlabForViz2C1(_TestGlabForModel):

  MODEL_CLASS = "viz2"

if __name__ == '__main__':
    unittest.main()
