# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

from glimpse.util.misc import GroupIterator, UngroupIterator
import unittest
import itertools

class TestGrouping(unittest.TestCase):

  def testGroup_nonFactorGroupSize(self):
    elements = range(10)
    group_size = 3
    expected = [ (0, 1, 2), (3, 4, 5), (6, 7, 8), (9,) ]
    actual = list(GroupIterator(elements, group_size))
    self.assertEqual(expected, actual)

  def testGroup_emptyList(self):
    self.assertEqual([], list(GroupIterator([], 1)))

  def testGroup_sizeIsOne(self):
    elements = range(10)
    group_size = 1
    expected = [ (x,) for x in elements ]
    actual = list(GroupIterator(elements, group_size))
    self.assertEqual(expected, actual)

  def testGroup_sizeIsFive(self):
    elements = range(10)
    group_size = 5
    expected = [ (0, 1, 2, 3, 4), (5, 6, 7, 8, 9) ]
    actual = list(GroupIterator(elements, group_size))
    self.assertEqual(expected, actual)

  def testGroupUngroup(self):
    elements = range(10)
    group_size = 2
    actual = list(UngroupIterator(GroupIterator(elements, group_size)))
    self.assertEqual(elements, actual)

  def testUngroup(self):
    elements = range(10)
    groups = [ [0, 1, 2, 3, 4], [5, 6, 7, 8, 9] ]
    self.assertEqual(elements, list(UngroupIterator(groups)))
    groups = [ [x] for x in elements ]
    self.assertEqual(elements, list(UngroupIterator(groups)))
    groups = [ [0, 1, 2], [3, 4, 5], [6, 7, 8], [9] ]
    self.assertEqual(elements, list(UngroupIterator(groups)))

  def testUngroup_emptyList(self):
    self.assertEqual([], list(UngroupIterator([])))

if __name__ == '__main__':
  unittest.main()
