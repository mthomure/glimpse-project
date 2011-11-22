# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.executors.executor import BasicExecutor, StaticMap, DynamicMap, \
    DynamicMapRequestHandler
import unittest

class TestBasicExecutor(unittest.TestCase):

  # test that single Put()/Get() call pair works
  def test_putGet(self):
    request_handler = lambda x: x * 100
    ex = BasicExecutor(request_handler)
    ex.Put(2)
    self.assertEqual(200, ex.Get())

class TestStaticMap(unittest.TestCase):

  # test that map works, modulo a reordering
  def test_basicExecutor(self):
    request_handler = lambda x: x * 100
    ex = BasicExecutor(request_handler)
    xs = range(10)
    actual = sorted(StaticMap(ex, xs))
    expected = map(request_handler, xs)
    self.assertEqual(expected, actual)

class TestDynamicMap(unittest.TestCase):

  def test(self):
    executor = BasicExecutor(DynamicMapRequestHandler)
    function = lambda x: x * 100
    xs = range(10)
    expected = map(function, xs)
    actual = sorted(DynamicMap(executor, function, xs))
    self.assertEqual(expected, actual)

  def test_largeGroup(self):
    executor = BasicExecutor(DynamicMapRequestHandler)
    function = lambda x: x * 100
    xs = range(100)
    expected = map(function, xs)
    actual = sorted(DynamicMap(executor, function, xs))
    self.assertEqual(expected, actual)

  def test_nonFactorGroupSize(self):
    executor = BasicExecutor(DynamicMapRequestHandler)
    function = lambda x: x * 100
    xs = range(10)
    expected = map(function, xs)
    actual = sorted(DynamicMap(executor, function, xs, group_size = 3))
    self.assertEqual(expected, actual)

if __name__ == '__main__':
  unittest.main()
