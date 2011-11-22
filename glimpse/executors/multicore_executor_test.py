# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.executors.executor import StaticMap, DynamicMap, \
    DynamicMapRequestHandler
from glimpse.executors.multicore_executor import MulticoreExecutor
import logging
import unittest

class TestFunction(object):

  def __call__(self, x):
    return x * 100

class TestMulticoreExecutor(unittest.TestCase):

  def helper(self, xs, callback, num_processes = None):
    ex = MulticoreExecutor(callback, num_processes = num_processes)
    actual = sorted(StaticMap(ex, xs))
    del ex
    expected = map(callback, xs)
    self.assertEqual(expected, actual)

  # test that map works, modulo a reordering, for a single process
  def test_singleProcess(self):
    callback = lambda x: x * 100
    xs = range(10)
    self.helper(xs, callback, num_processes = 1)

  # test that result is correct for multiple processes
  def test_multipleProcesses(self):
    callback = lambda x: x * 100
    xs = range(10)
    self.helper(xs, callback, num_processes = 2)

  # test that Put/Get() pairs work as expected
  def test_putGet(self):
    callback = lambda x: x * 100
    ex = MulticoreExecutor(callback, num_processes = 1)
    ex.Put(2)
    self.assertEqual(200, ex.Get())

  # test that map is correct for single process
  def test_map_singleProcess(self):
    callback = lambda x: x * 100
    ex = MulticoreExecutor(callback, num_processes = 1)
    xs = range(10)
    actual = sorted(StaticMap(ex, xs))
    expected = map(callback, xs)
    self.assertEqual(expected, actual)

  # test that map is correct for multiple processes
  def test_map_multipleProcesses(self):
    callback = lambda x: x * 100
    ex = MulticoreExecutor(callback, num_processes = 4)
    xs = range(10)
    actual = sorted(StaticMap(ex, xs))
    expected = map(callback, xs)
    self.assertEqual(expected, actual)

  # test that map can be called repeatedly without error
  def test_map_remap(self):
    callback = lambda x: x * 100
    ex = MulticoreExecutor(callback, num_processes = 4)
    xs = range(10)
    actual = sorted(StaticMap(ex, xs))
    expected = map(callback, xs)
    self.assertEqual(expected, actual)
    xs = range(20, 30)
    actual = sorted(StaticMap(ex, xs))
    expected = map(callback, xs)
    self.assertEqual(expected, actual)

  # test that all processes are used
  def test_map_usingAllProcesses(self):
    def callback(x):
      import os
      return os.getpid()
    num_processes = 8
    ex = MulticoreExecutor(callback, num_processes = num_processes)
    xs = range(num_processes * 100)
    num_used_processes = len(set(StaticMap(ex, xs)))
    self.assertEqual(num_processes, num_used_processes)


class TestMulticoreExecutor(unittest.TestCase):

  def test_dynamicMap(self):
    executor = MulticoreExecutor(DynamicMapRequestHandler, num_processes = 4)
    function = TestFunction()  # lambda function isn't picklable, so use object
    xs = range(10)
    expected = map(function, xs)
    actual = sorted(DynamicMap(executor, function, xs))
    self.assertEqual(expected, actual)

  def _test_basicExecutor(self):
    from glimpse.executors.executor import BasicExecutor
    executor = BasicExecutor(DynamicMapRequestHandler)
    function = TestFunction()
    xs = range(10)
    expected = map(function, xs)
    actual = sorted(DynamicMap(executor, function, xs))
    self.assertEqual(expected, actual)

if __name__ == '__main__':

  # Uncomment the following for debugging messages.
  #logging.getLogger().setLevel(logging.INFO)

  unittest.main()
