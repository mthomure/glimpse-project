import unittest

from .state import *
from .node import *

n1 = Node("n1")
n2 = Node("n2", depends = [n1])
n3 = Node("n3", depends = [n2])

n1_state = 'n1-state'
n2_state = 'n2-state'
n3_state = 'n3-state'

class State2(State):
  pass

def FillState():
  s = State()
  s[n1.id_] = n1_state
  s[n2.id_] = n2_state
  return s

class TestState(unittest.TestCase):

  def testInit(self):
    ref_state = FillState()
    state = State(((n1, n1_state), (n2, n2_state)))
    self.assertEqual(state, ref_state)
    state = State({n1 : n1_state, n2 : n2_state})
    self.assertEqual(state, ref_state)
    state = State(n1 = n1_state, n2 = n2_state)  # using id directly
    self.assertEqual(state, ref_state)

  def testGetSetItem(self):
    s = FillState()
    self.assertEqual(s[n1], n1_state)
    s[n3] = n3_state
    self.assertEqual(s[n3], n3_state)
    self.assertEqual(s[n3.id_], n3_state)
    s = FillState()
    s[n3.id_] = n3_state
    self.assertEqual(s[n3], n3_state)
    self.assertEqual(s[n3.id_], n3_state)

  def testDelItem(self):
    s = FillState()
    del s[n1]
    with self.assertRaises(KeyError):
      s[n1]
    with self.assertRaises(KeyError):
      s[n1.id_]
    self.assertEqual(s[n2], n2_state)

  def testContains(self):
    s = FillState()
    self.assertIn(n1, s)
    self.assertIn(n1.id_, s)
    self.assertNotIn(n3, s)
    self.assertNotIn(n3.id_, s)

  def testGet(self):
    s = FillState()
    self.assertEqual(s.get(n1), n1_state)

  def testUpdate(self):
    s = State()
    s.update({n1 : n1_state, n2 : n2_state})
    self.assertEqual(s, FillState())
    s = State()
    s.update(((n1, n1_state), (n2, n2_state)))
    self.assertEqual(s, FillState())

  def testRepr(self):
    state = State(((n1, n1_state), (n2, n2_state)))
    self.assertIsNotNone(repr(state))

  def testStr(self):
    state = State(((n1, n1_state), (n2, n2_state)))
    self.assertIsNotNone(str(state))

  def testCopy(self):
    state = State2(((n1, n1_state), (n2, n2_state)))
    self.assertEqual(state, state.copy())

  def testLen(self):
    state = State2(((n1, n1_state), (n2, n2_state)))
    self.assertEqual(len(state), 2)

if __name__ == '__main__':
  unittest.main()
