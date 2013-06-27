# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import copy
from decorator import decorator
import inspect
from pprint import pformat
from glimpse.util.callback import Callback

__all__ = [
    'node_builder',
    'DependencyError',
    'DataFlow',
    'BuildNode',
    ]

def _func_wrapper(f, *args, **kw):
  return f(*args, **kw)

def node_builder(node):
  """This decorator registers the method for the given node."""
  if inspect.isfunction(node):
    # If `node` is a function, user must have forgotten the decorator argument.
    raise ValueError("@node_builder decorator takes an argument")
  def dec(f):
    f._flow_node = node
    return decorator(_func_wrapper, f)
  return dec

class DependencyError(Exception):
  """Thrown when a requested node has a non-computable dependency."""

  def __init__(self, node):
    self.node = node

  def __str__(self):
    return "Node (%s) could not be computed due to unmet dependencies" % \
        pformat(self.node)

  def __repr__(self):
    return "%s(node=%s)" % (type(self).__name__, pformat(self.node))

class DataFlow(object):
  """A dataflow is a graph with data as nodes and computations as edges."""

  def __init__(self):
    self._callbacks = dict()
    for name, method in inspect.getmembers(self):
      if (hasattr(method, 'undecorated') and
          hasattr(method.undecorated, '_flow_node')):
        self.Register(method.undecorated._flow_node, method)

  def Register(self, node, f):
    """Add a callback to compute a value for a node.

    Note: be careful of passing a lambda function for ``f``, as this will likely
    cause an error if serialized.

    """
    num_f_args = len(inspect.getargspec(f)[0])
    if hasattr(f, 'im_self') and f.im_self is not None:
      # first argument is already bound to object instance
      num_f_args -= 1
    if node.depends is None:
      num_args = 0
    else:
      num_args = len(node.depends)
    if num_f_args != num_args:
      raise ValueError("Callback must take %d arguments, but instead takes %d" \
          % (num_args, num_f_args))
    if f is None:
      del self._callbacks[node]
    else:
      # Callbacks are wrapped to allow dataflow to be pickled
      f = Callback(f)
      self._callbacks[node] = f

  def _Dispatch(self, node, args):
    """Compute the value of a single node."""
    if node not in self._callbacks:
      raise DependencyError(node)
    return self._callbacks[node](*args)

  def _BuildRecursive(self, node, state):
    """Compute node and dependencies, storing results in `state`."""
    # Short-circuit computation if data exists
    if node in state:
      return state[node]
    # Compute any dependencies
    if node.depends is None:
      args = ()
    else:
      args = [ self._BuildRecursive(n, state) for n in node.depends ]
    # Compute the output node
    value = self._Dispatch(node, args)
    state[node] = value
    return value

def BuildNode(flow, nodes, in_state):
  out_state = copy.copy(in_state)
  if not hasattr(nodes, '__len__'):
    nodes = (nodes,)
  for node in nodes:
    flow._BuildRecursive(node, out_state)
  return out_state
