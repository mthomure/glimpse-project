
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import transform as core
import unittest

class RegionMapper(unittest.TestCase):
  options = core.ExpandOptions(dict(retina_enabled = True, retina_kwidth = 15, s1_kwidth = 11, s1_scaling = 2, c1_kwidth = 5, c1_scaling = 2, s2_kwidth = 7, s2_scaling = 2, c2_kwidth = 3, c2_scaling = 2))
  rm = core.RegionMapper(options)

  def testRetinaToImage(self):
    self.assertEqual(slice(0, 15),
                     self.rm.MapRetinaToImage(slice(0, 1)))

  def testS1ToImage(self):
    self.assertEqual(slice(0, 25),
                     self.rm.MapS1ToImage(slice(0, 1)))

  def testC1ToImage(self):
    self.assertEqual(slice(0, 33),
                     self.rm.MapC1ToImage(slice(0, 1)))

  def testS2ToImage(self):
    self.assertEqual(slice(0, 57),
                     self.rm.MapS2ToImage(slice(0, 1)))

  def testC2ToImage(self):
    self.assertEqual(slice(0, 73),
                     self.rm.MapC2ToImage(slice(0, 1)))

if __name__ == '__main__':
  unittest.main()
