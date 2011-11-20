
from glimpse.executors.executor import ExecutorMap
from glimpse.executors.multicore_executor import MulticoreExecutor
import logging
import unittest

class TestMulticoreExecutor(unittest.TestCase):

  def helper(self, xs, callback, num_processes = None):
    ex = MulticoreExecutor(callback, num_processes = num_processes)
    actual = sorted(ExecutorMap(ex, xs))
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
    actual = sorted(ExecutorMap(ex, xs))
    expected = map(callback, xs)
    self.assertEqual(expected, actual)

  # test that map is correct for multiple processes
  def test_map_multipleProcesses(self):
    callback = lambda x: x * 100
    ex = MulticoreExecutor(callback, num_processes = 4)
    xs = range(10)
    actual = sorted(ExecutorMap(ex, xs))
    expected = map(callback, xs)
    self.assertEqual(expected, actual)

  # test that map can be called repeatedly without error
  def test_map_remap(self):
    callback = lambda x: x * 100
    ex = MulticoreExecutor(callback, num_processes = 4)
    xs = range(10)
    actual = sorted(ExecutorMap(ex, xs))
    expected = map(callback, xs)
    self.assertEqual(expected, actual)
    xs = range(20, 30)
    actual = sorted(ExecutorMap(ex, xs))
    expected = map(callback, xs)
    self.assertEqual(expected, actual)

  # test that all processes are used
  def test_map_usingAllProcesses(self):
    def callback(x):
      import os
      return os.getpid()
    num_processes = 8
    ex = MulticoreExecutor(callback, num_processes = num_processes)
    xs = range(num_processes * 10)
    num_used_processes = len(set(ExecutorMap(ex, xs)))
    self.assertEqual(num_processes, num_used_processes)

if __name__ == '__main__':

  # Uncomment the following for debugging messages.
  #logging.getLogger().setLevel(logging.INFO)

  unittest.main()
