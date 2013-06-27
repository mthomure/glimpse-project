# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from pprint import pformat

__all__ = [
    'Node'
    ]

class Node(object):
  id_ = 0
  depends = []

  def __init__(self, id_, depends = None):
    if not (depends is None or hasattr(depends, '__len__')):
      raise ValueError("Dependencies argument must be a list")
    self.id_, self.depends = id_, depends

  def __str__(self):
    return "%s(%s)" % (type(self).__name__, self.id_)

  def __repr__(self):
    return "%s(%s)" % (type(self).__name__, ", ".join("%s=%s" % (k,
        pformat(getattr(self, k))) for k in ("id_", "depends")))

  def __hash__(self):
    return hash(self.id_)

  def __cmp__(self, other):
    # Note that this is only useful for comparing nodes within a single
    # dataflow, since it only considers the `id_` attribute.
    return cmp(self.id_, other.id_)
