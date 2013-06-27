# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from .node import Node

__all__ = [
    'State'
    ]

class State(object):
  """A mapping that captures the state of a data flow.

  Keys can be :class:`Node` instances or the value of its :attr:`Node.id_`
  attribute.

  """

  # We delegate from dict rather than extend the dict class. This was done to
  # work around a bug in IPython 0.13.1, in which any dict subclass is pickled
  # as a dict. (See IPython/zmq/serialize.py, line 72.)

  def __init__(self, elements = None, **kw):
    # Keys of kw are guaranteed to be id_s, not Node (since they can't be
    # expressions).
    self._data = dict(**kw)
    if elements is not None:
      if hasattr(elements, 'items'):
        elements = elements.items()
      for k, v in elements:
        if k not in kw:
          self[k] = v

  def __len__(self):
    return len(self._data)

  def __getitem__(self, name):
    """Lookup activation for a given node.

    :param name: Node id_ifier.
    :type name: scalar or :class:`Node`

    """
    if isinstance(name, Node):
      name = name.id_
    return self._data[name]

  def __setitem__(self, name, value):
    if isinstance(name, Node):
      name = name.id_
    self._data[name] = value

  def __delitem__(self, name):
    if isinstance(name, Node):
      name = name.id_
    del self._data[name]

  def __contains__(self, name):
    if isinstance(name, Node):
      name = name.id_
    return name in self._data

  def __iter__(self):
    return iter(self._data)

  def __eq__(self, other):
    return type(self) == type(other) and self.items() == other.items()

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return "%s(%r)" % (type(self).__name__, self._data)

  def __str__(self):
    return "%s(%s)" % (type(self).__name__, ", ".join(self.keys()))

  def keys(self):
    return self._data.keys()

  def items(self):
    return self._data.items()

  def values(self):
    return self._data.values()

  def clear(self):
    self._data.clear()

  def copy(self):
    return self.__class__(self._data.copy())

  def get(self, name, default_value = None):
    if isinstance(name, Node):
      name = name.id_
    return self._data.get(name, default_value)

  def update(self, e, **f):
    if hasattr(e, 'items'):
      e = e.items()
    for k, v in e:
      self[k] = v
    for k, v in f.items():
      self[k] = v
