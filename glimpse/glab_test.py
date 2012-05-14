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

EXAMPLE_CORPUS = pjoin(os.path.dirname(glab.__file__), '..', 'rc',
    'example-corpus')

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

  def testImprintS2Prototypes(self):
    # Look up the example corpus location relative to the Glimpse code.
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.ImprintS2Prototypes(10)
    e = glab.GetExperiment()
    self.assertEqual(len(e.model.s2_kernels), 10)

  def testMakeUniformRandomS2Prototypes(self):
    glab.MakeUniformRandomS2Prototypes(10)
    e = glab.GetExperiment()
    self.assertEqual(len(e.model.s2_kernels), 10)

  def testMakeShuffledRandomS2Prototypes(self):
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.MakeShuffledRandomS2Prototypes(10)
    e = glab.GetExperiment()
    self.assertEqual(len(e.model.s2_kernels), 10)

  def testMakeHistogramRandomS2Prototypes(self):
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.MakeHistogramRandomS2Prototypes(10)
    e = glab.GetExperiment()
    self.assertEqual(len(e.model.s2_kernels), 10)

  def testMakeNormalRandomS2Prototypes(self):
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.MakeNormalRandomS2Prototypes(10)
    e = glab.GetExperiment()
    self.assertEqual(len(e.model.s2_kernels), 10)

  def testRunSvm(self):
    glab.SetCorpus(EXAMPLE_CORPUS)
    glab.ImprintS2Prototypes(10)
    glab.RunSvm()
    e = glab.GetExperiment()
    self.assertNotEqual(e.train_results['accuracy'], None)
    self.assertNotEqual(e.test_results['accuracy'], None)

if __name__ == '__main__':
    unittest.main()
