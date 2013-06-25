import os
import unittest

from glimpse.util.garray import toimage
from glimpse.models.ml import Model, Layer, Params
from glimpse.models.ml.param import SLayerOp
from glimpse.models.base import BuildLayer
from glimpse.pools import SinglecorePool
from glimpse.util.gtest import *

from .utils import *

def MakeCorpus(root, **subdirs):
  for cls, files in subdirs.items():
    os.mkdir(os.path.join(root, cls))
    for f in files:
      with open(os.path.join(root, cls, f), 'w') as fh:
        fh.write("1")

class CorpusTest(unittest.TestCase):

  def testReadCorpusDirs_multiclass(self):
    corpus = dict(cls_a = ("11.jpg", "12.jpg"),
        cls_b = ("21.jpg", "22.jpg", "23.jpg"))
    with TempDir() as root:
      MakeCorpus(root, **corpus)
      dirs = [ os.path.join(root, x) for x in corpus ]
      paths, labels = ReadCorpusDirs(dirs, DirReader())
      paths = map(os.path.basename, paths)
      paths_a = sorted(paths[:2])
      paths_b = sorted(paths[2:])
      self.assertSequenceEqual(paths_a, corpus["cls_a"])
      self.assertSequenceEqual(paths_b, corpus["cls_b"])
      self.assertTrue(np.all(labels == (0,0,1,1,1)))

  def testReadCorpusDirs_oneClass(self):
    corpus = dict(cls_a = ("11.jpg", "12.jpg"))
    with TempDir() as root:
      MakeCorpus(root, **corpus)
      dirs = [ os.path.join(root, x) for x in corpus ]
      paths, labels = ReadCorpusDirs(dirs, DirReader())
      paths = map(os.path.basename, paths)
      self.assertSequenceEqual(sorted(paths), corpus['cls_a'])
      self.assertTrue(np.all(labels == (0,0)))

  def testReadCorpusDirs_failsOnEmptyDir(self):
    corpus = dict(cls_a = tuple(), cls_b = ("21.jpg", "22.jpg"))
    with TempDir() as root:
      MakeCorpus(root, **corpus)
      dirs = [ os.path.join(root, x) for x in corpus ]
      with self.assertRaises(Exception):
        ReadCorpusDirs(dirs, DirReader())

  def testReadCorpusDirs_failsOnNoImages(self):
    with TempDir() as root:
      with self.assertRaises(Exception):
        ReadCorpusDirs([], DirReader())

  def testBalanceCorpus_balancedInput(self):
    labels = np.array([1, 1, 2, 2])
    mask = BalanceCorpus(labels)
    self.assertEqual(mask.sum(), 4)  # all instances should be used

  def testBalanceCorpus_unbalancedInput(self):
    labels = np.array([1, 1, 2, 2, 2])
    mask = BalanceCorpus(labels)
    self.assertTrue(np.all(mask == [True, True, True, True, False]))

  def testBalanceCorpus_unbalancedInput_withShuffle(self):
    labels = np.array([1, 1, 2, 2, 2])
    mask = BalanceCorpus(labels)
    self.assertTrue(np.all(mask[:2] == [True, True]))
    # we only know that two of the available class-2 instances should be used
    self.assertEqual(mask[2:].sum(), 2)

class FeatureBuildingTest(unittest.TestCase):

  def testExtractFeatures_oneImage(self):
    m = Model()
    img = toimage(np.random.random((100, 100)) * 255)
    s = BuildLayer(m, Layer.C1, m.MakeState(img))
    fs = ExtractFeatures(Layer.S1, [s])
    self.assertEqual(fs.shape[0], 1)
    self.assertGreater(fs.shape[1], 0)

  def testExtractFeatures_multipleImages(self):
    m = Model()
    img1 = toimage(np.random.random((100, 100)) * 255)
    img2 = toimage(np.random.random((100, 100)) * 255)
    s1 = BuildLayer(m, Layer.C1, m.MakeState(img1))
    s2 = BuildLayer(m, Layer.C1, m.MakeState(img2))
    fs = ExtractFeatures(Layer.S1, [s1, s2])
    self.assertEqual(fs.shape[0], 2)
    self.assertGreater(fs.shape[1], 0)

  def testExtractFeatures_multipleLayers(self):
    m = Model()
    img = toimage(np.random.random((100, 100)) * 255)
    s = BuildLayer(m, Layer.C1, m.MakeState(img))
    fs = ExtractFeatures((Layer.S1, Layer.C1), [s])
    self.assertEqual(fs.shape[0], 1)
    self.assertGreater(fs.shape[1], 0)

if __name__ == '__main__':
  unittest.main()
