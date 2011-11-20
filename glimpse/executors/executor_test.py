
from glimpse.executors.executor import BasicExecutor, ExecutorMap
import unittest

class TestBasicExecutor(unittest.TestCase):

  # test that map works, modulo a reordering
  def test_map(self):
    callback = lambda x: x * 100
    ex = BasicExecutor(callback)
    xs = range(10)
    actual = sorted(ExecutorMap(ex, xs))
    expected = map(callback, xs)
    self.assertEqual(expected, actual)

  # test that single Put()/Get() call pair works
  def test_putGet(self):
    callback = lambda x: x * 100
    ex = BasicExecutor(callback)
    ex.Put(2)
    self.assertEqual(200, ex.Get())

if __name__ == '__main__':
  unittest.main()
