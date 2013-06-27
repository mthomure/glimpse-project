# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import unittest

from glimpse.util.data import Data

import unittest

class Data1(Data):
  k1 = None

class EmptyData(Data):
  pass

class DataInner():
  def __init__(self):
    self.init = True

class DataOuter(Data):
  inner = DataInner()

class DataWithCollection(Data):
  list_ = [1,2]

class DataTest(unittest.TestCase):

  def testNestedData(self):
    # ensure that nested objects are correctly initialized
    obj = DataOuter()
    self.assertNotEqual(id(obj.inner), DataOuter.inner)
    self.assertTrue(hasattr(obj.inner, 'init'))

  def testDeepCopy(self):
    # ensure that collection variables are deep copied
    obj1 = DataWithCollection()
    obj1.list_.append(3)
    obj2 = DataWithCollection()
    obj2.list_.append(4)
    self.assertNotEqual(obj1.list_, obj2.list_)
    self.assertNotEqual(obj1.list_, DataWithCollection.list_)

  def testSetItem(self):
    d = Data1()
    d['k1'] = 1
    self.assertEqual(d.__dict__['k1'], 1)

  def testSetItem_newKey(self):
    d = Data1()
    d['k2'] = 2
    self.assertEqual(d.__dict__['k2'], 2)

  def testSetAttr(self):
    d = Data1()
    d.k1 = 1
    self.assertEqual(d.__dict__['k1'], 1)

  def testSetAttr_newKey(self):
    d = Data1()
    with self.assertRaises(AttributeError):
      d.k2 = 2

  def testDelAttr(self):
    d = Data1()
    d.__dict__['k2'] = 2
    del d.k2
    self.assertNotIn('k2', d.__dict__)

  def testDelAttr_withDefault(self):
    d = Data1()
    d.__dict__['k1'] = 1
    del d.k1
    self.assertEqual(d.__dict__['k1'], None)

  def testContains(self):
    d = Data1()
    self.assertIn('k1', d)
    self.assertNotIn('k2', d)

  def testPickle(self):
    import cPickle as pickle
    d = Data1()
    d.__dict__['k1'] = 1
    d.__dict__['k2'] = 2
    d2 = pickle.loads(pickle.dumps(d, protocol=2))
    self.assertEqual(d, d2)

  def testRepr(self):
    # should not cause exception
    d = Data1()
    d.__dict__['k1'] = 1
    d.__dict__['k2'] = 2
    repr(d)

  def testInit(self):
    # empty
    d = Data1()
    self.assertEqual(d.__dict__, {'k1':None})
    # with fields
    fields = {'k1':1, 'k2':2}
    d = Data1(fields)
    self.assertEqual(d.__dict__, fields)
    # with kw
    d = Data1(**fields)
    self.assertEqual(d.__dict__, fields)

  def testEqual(self):
    d1 = Data1()
    d1.__dict__['k1'] = 1
    d1.__dict__['k2'] = 2
    d2 = Data1()
    d2.__dict__['k1'] = 1
    d2.__dict__['k2'] = 2
    self.assertEqual(d1, d2)

  def testIter(self):
    d = Data1()
    d.__dict__['k1'] = 1
    d.__dict__['k2'] = 2
    self.assertSequenceEqual(sorted(iter(d)), ('k1','k2'))
    self.assertSequenceEqual(tuple(iter(EmptyData())), ())

  def testLength(self):
    d = Data1()
    d.__dict__['k1'] = 1
    d.__dict__['k2'] = 2
    self.assertEqual(len(d), 2)
    self.assertEqual(len(EmptyData()), 0)

  def testDir(self):
    d = Data1()
    d.__dict__['k1'] = 1
    d.__dict__['k2'] = 2
    keys = [k for k in dir(d) if not k.startswith('_')]
    self.assertSequenceEqual(sorted(keys), ('k1','k2'))
    keys = [k for k in dir(EmptyData()) if not k.startswith('_')]
    self.assertSequenceEqual(keys, ())

if __name__ == '__main__':
    unittest.main()
