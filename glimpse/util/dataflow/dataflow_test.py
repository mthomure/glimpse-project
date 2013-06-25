import cPickle as pickle
import unittest

from .dataflow import *
from .node import Node
from .state import State

n1 = Node("n1")
n2_1 = Node("n2_1", depends = [n1])
n2_2 = Node("n2_2", depends = [n1])
n3 = Node("n3", depends = [n2_1, n2_2])
n4 = Node("n4")
# Can't be a lambda function
def Compute2_1(o1): return ["2_1"] + o1
def Compute2_2(o1): return ["2_2"] + o1
def Compute3(o2_1, o2_2): return ["3"] + o2_1 + o2_2

class MyFlow(DataFlow):

  @node_builder(n2_1)
  def BuildN2_1(self, d1):
    return Compute2_1(d1)

  @node_builder(n2_2)
  def BuildN2_2(self, d1):
    return Compute2_2(d1)

  @node_builder(n3)
  def BuildN3(self, d2_1, d2_2):
    return Compute3(d2_1, d2_2)

class DataFlowTest(unittest.TestCase):

  def testDecorator(self):
    d = MyFlow()
    s = BuildNode(d, n3, State({n1 : ["1"]}))
    for node in (n1, n2_1, n2_2, n3):
      self.assertIn(node, s)
    self.assertEqual(s[n3], ["3", "2_1", "1", "2_2", "1"])

  def testBuildNode_multipleNodes(self):
    # pass multiple nodes, check that each are built
    d = MyFlow()
    s = BuildNode(d, (n2_1, n2_2), State({n1 : ["1"]}))
    for node in (n1, n2_1, n2_2):
      self.assertIn(node, s)
    self.assertEqual(s[n2_1], ["2_1", "1"])
    self.assertEqual(s[n2_2], ["2_2", "1"])

  def testBuildNode_badNode(self):
    d = MyFlow()
    with self.assertRaises(DependencyError):
      BuildNode(d, n4, State())

  def testEndToEnd(self):
    d = DataFlow()
    d.Register(n2_1, Compute2_1)
    d.Register(n2_2, Compute2_2)
    d.Register(n3, Compute3)
    s = BuildNode(d, n3, State({n1 : ["1"]}))
    for node in (n1, n2_1, n2_2, n3):
      self.assertIn(node, s)

  def testBottomNodeGenerator(self):
    d = DataFlow()
    d.Register(n1, lambda: 1)  # callback returns constant value
    s = BuildNode(d, n1, State())
    self.assertIn(n1, s)
    self.assertEqual(s[n1], 1)

  def testSerializable_noDecorator(self):
    d = DataFlow()
    d.Register(n2_1, Compute2_1)
    d.Register(n2_2, Compute2_2)
    d.Register(n3, Compute3)
    d2 = pickle.loads(pickle.dumps(d, protocol = 2))
    self.assertSequenceEqual(d._callbacks.keys(), d2._callbacks.keys())

  def testSerializable_withDecorator(self):
    d1 = MyFlow()
    d2 = pickle.loads(pickle.dumps(d1, protocol = 2))
    # The same set of nodes should be in the flow:
    self.assertSetEqual(set(d1._callbacks.keys()), set(d2._callbacks.keys()))
    # However, callbacks won't be equal, since they'll be bound to different
    # objects.
    for k in d1._callbacks.keys():
      f1 = d1._callbacks[k].f
      f2 = d2._callbacks[k].f
      # Just compare unbound method.
      self.assertEqual(f1.im_func, f2.im_func)

  def testRegister_badCallback(self):
    d = DataFlow()
    with self.assertRaises(ValueError):
      d.Register(n2_1, toofew_callback)
    with self.assertRaises(ValueError):
      d.Register(n2_1, toomany_callback)

def toofew_callback(): return "exval"
def toomany_callback(x, y): return "exval"

if __name__ == '__main__':
  unittest.main()
