
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

from layer_mapping import RegionMapper
from params import Params
import unittest

class TestRegionMapper(unittest.TestCase):
  params = Params()
  params.retina_enabled = True
  params.retina_kwidth = 15
  params.s1_kwidth = 11
  params.s1_scaling = 2
  params.c1_kwidth = 5
  params.c1_scaling = 2
  params.s2_kwidth = 7
  params.c2_kwidth = 3
  params.c2_scaling = 2
  rm = RegionMapper(params)

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
