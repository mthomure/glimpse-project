from glimpse.util.gtest import *
from glimpse.util.option import *

class ImmutableAttributeSetTest(unittest.TestCase):

  def testEmptySet(self):
    aset = ImmutableAttributeSet()
    with self.assertRaises(AttributeError):
      aset.newkey = 1

  def testInit(self):
    aset = ImmutableAttributeSet(key1 = 1, key2 = 2)
    self.assertEqual(aset.key1, 1)
    self.assertEqual(aset.key2, 2)

  def testGetSet(self):
    aset = ImmutableAttributeSet(key1 = None)
    aset.key1 = 10
    self.assertEqual(aset.key1, 10)
    with self.assertRaises(Exception):
      del aset.key

if __name__ == '__main__':
  unittest.main()
