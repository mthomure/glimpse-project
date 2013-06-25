import unittest

from .learn import *

class LearnTests(unittest.TestCase):

  def testChooseTrainingSet_singleClass(self):
    labels = np.ones((100,))
    mask = ChooseTrainingSet(labels, 0.5)
    self.assertEqual(mask.sum(), 50)
    mask = ChooseTrainingSet(labels, 0.01)
    self.assertEqual(mask.sum(), 1)
    mask = ChooseTrainingSet(labels, 0.90)
    self.assertEqual(mask.sum(), 90)

  def testChooseTrainingSet_multiClass(self):
    labels = np.array([1]*100 + [2]*200)
    mask = ChooseTrainingSet(labels, 0.5)
    self.assertEqual(mask[:100].sum(), 50)
    self.assertEqual(mask[100:].sum(), 100)
    mask = ChooseTrainingSet(labels, 0.01)
    self.assertEqual(mask[:100].sum(), 1)
    self.assertEqual(mask[100:].sum(), 2)
    mask = ChooseTrainingSet(labels, 0.99)
    self.assertEqual(mask[:100].sum(), 99)
    self.assertEqual(mask[100:].sum(), 198)

  def testChooseTrainingSet_valueErrorOnSingletonClass(self):
    with self.assertRaises(ValueError):
      ChooseTrainingSet(np.array([1, 1, 2]), 0.5)

  def testChooseTrainingSet_valueErrorOnBadSplitSize(self):
    labels = np.array([1, 1])
    with self.assertRaises(ValueError):
      ChooseTrainingSet(labels, -1)

  def testTrainTestClassifier(self):
    features = np.random.random((4, 4))
    labels = np.array([1,1,2,2])
    clf = TrainClassifier(features, labels)
    self.assertIsNotNone(clf)
    np.random.shuffle(features)
    np.random.shuffle(labels)
    acc,_ = ScoreClassifier(features, labels, clf)
    self.assertGreaterEqual(acc, 0)
    self.assertLessEqual(acc, 1)

  def testCrossValidateClassifier(self):
    features = np.random.random((40, 4))
    labels = np.array([1]*20 + [2]*20)
    np.random.shuffle(features)
    np.random.shuffle(labels)
    acc_list = CrossValidateClassifier(features, labels)
    self.assertGreater(len(acc_list), 0)
    for acc in acc_list:
      self.assertGreaterEqual(acc, 0)
      self.assertLessEqual(acc, 1)

if __name__ == '__main__':
  unittest.main()



"""

ResolveScoreFunc
  - passes function through
  - handles None
  - handles string

"""
