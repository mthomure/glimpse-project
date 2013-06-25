import unittest

from .layer import *

class L(Layer):

  L1 = LayerSpec("l1", "layer-1")
  L2 = LayerSpec("l2", "layer-2", [L1])

class TestLayer(unittest.TestCase):

  def testFromId(self):
    self.assertEqual(L.FromId(L.L1.id_), L.L1)
    self.assertEqual(L.FromId(L.L2.id_), L.L2)

  def testFromName(self):
    self.assertEqual(L.FromName(L.L1.name), L.L1)
    self.assertEqual(L.FromName(L.L2.name), L.L2)

  def testAllLayers(self):
    self.assertEqual(set(L.AllLayers()), set((L.L2, L.L1)))

  def testIsSublayer(self):
    self.assertTrue(L.IsSublayer(L.L1, L.L2))
    self.assertFalse(Layer.IsSublayer(L.L2, L.L1))

  def testTopLayer(self):
    self.assertEqual(L.TopLayer(), L.L2)

if __name__ == '__main__':
  unittest.main()
