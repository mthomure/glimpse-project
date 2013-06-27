# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import unittest
import cPickle as pickle

from .callback import *

class C(object):

  def __init__(self, arg = None):
    self.arg = arg

  def __eq__(self, other):
    return type(self) == type(other) and self.arg == other.arg

  def c_method(self, *args, **kw):
    return "C.method", self.arg, args, kw

  @classmethod
  def c_class_method(cls, *args, **kw):
    return "C.class_method", args, kw

def test_function(*args, **kw):
  return "test_function", args, kw

def PicklingCallback(f, *args, **kw):
  cb = Callback(f, *args, **kw)
  return pickle.loads(pickle.dumps(cb, protocol = 2))

def _test(tester, f, callback_func = None):
  if callback_func is None:
    callback_func = Callback
  tester.assertEqual(f, callback_func(f).f)
  tester.assertEqual(f(), callback_func(f)())
  tester.assertEqual(f(1, 2), callback_func(f)(1, 2))
  tester.assertEqual(f(1, 2), callback_func(f, 1, 2)())
  tester.assertEqual(f(1, 2, 3, 4), callback_func(f, 1, 2)(3, 4))
  tester.assertEqual(f(key1 = 1, key2 = 2), callback_func(f)(key1 = 1,
      key2 = 2))
  tester.assertEqual(f(key1 = 1, key2 = 2), callback_func(f, key1 = 1,
      key2 = 2)())
  tester.assertEqual(f(key1 = 1, key2 = 2, key3 = 3, key4 = 4), callback_func(f,
      key1 = 1, key2 = 2)(key3 = 3, key4 = 4))
  # Make sure repr works with args and keywords
  tester.assertIsNotNone(repr(callback_func(f, 1, 2, key1 = 1, key2 = 2)))

class CallbackTest(unittest.TestCase):

  def testCallFunction(self):
    _test(self, test_function)

  def testCallBoundMethod(self):
    c = C('obj-arg')
    _test(self, c.c_method)

  def testCallClassMethod(self):
    _test(self, C.c_class_method)

  def testLambdaRaisesError(self):
    with self.assertRaises(pickle.PicklingError):
      Callback(lambda: None).__getstate__()

  def testUnboundMethodRaisesError(self):
    with self.assertRaises(pickle.PicklingError):
      Callback(C.c_method).__getstate__()

  def testPickleFunction(self):
    _test(self, test_function, PicklingCallback)

  def testPickleBoundMethod(self):
    c = C('notnone')
    _test(self, c.c_method, PicklingCallback)

  def testPickleClassMethod(self):
    _test(self, C.c_class_method, PicklingCallback)

if __name__ == '__main__':
    unittest.main()
