# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import copy
import pprint

class Data(object):
  """An extended dictionary interface with attribute access to key-value pairs.

  Properties of the Data object include:
    - Existing keys are accessible as attributes or dictionary items.
    - Default values can be specified as class variables. These variables are
      reset to their default value upon "del"etion, while other keys are removed
      upon deletion.
    - New keys can be added as dictionary items only. Read or write access to an
      unknown key via its attribute will raise an exception.

  Example:
  >>> class Data(): pass  # new container with no defaults
  >>> data = A(u1=1)  # create and add new key
  >>> print data.u1  # access key as attribute
  >>> print data['u1']  # access key as item
  >>> data['u2'] = 2  # add new key
  >>> del data.u1

  Example:
  >>> class B(Data): key = None  # new container with single default
  >>> data = A()  # create without adding keys
  >>> print data.key  # access key as attribute, which still has default value
  >>> data2 = A(key = 2)  # create and override default value
  >>> del data2.key  # reset to default value
  >>> assert data.key == data2.key

  Benefits of the Data object include:
    - Improved access to an arbitrary set of key-value pairs (i.e., attribute
      rather than item access).
    - Exceptions on key misspellings via attributes.
    - Encourages use of simple data objects that are changed infrequently,
      rather than full OOP objects that may change quickly during development.
      This results in more stable pickled files that are readable for a longer
      period of time. Additionally, data is pickled as a dictionary, so even
      changes in the Data object will not prevent unpickling of past results.

  """

  def __init__(self, fields=None, **kw):
    """Create new Data object, and initialize given fields.

    :params dict fields: Dictionary of fields to initialize.
    :params kw: Fields to initialize.

    Note that inputs are applied after fields are initialized from defaults.

    """
    # apply defaults from class
    cls = type(self)
    for k in cls.__dict__:
      if k.startswith('_'):
        continue
      self.__dict__[k] = copy.deepcopy(getattr(cls,k))
    # apply updates from caller
    if fields is not None:
      self.__dict__.update(fields)
    self.__dict__.update(kw)

  def __deepcopy__(self, memo):
    newone = type(self)()
    for k,v in self.__dict__.items():
      if not k.startswith('_'):
        v = copy.deepcopy(v, memo)
      newone.__dict__[k] = v
    return newone

  def __contains__(self, k):
    return k in self.__dict__

  def __delattr__(self, k):
    try:
      self.__delitem__(k)
    except KeyError:
      raise AttributeError("'%s' object has no attribute '%s'" % (
          type(self).__name__, k))

  def __delitem__(self, k):
    cls = type(self)
    if hasattr(cls, k):
      self.__dict__[k] = copy.deepcopy(getattr(type(self), k))  # reinitialize
    else:
      del self.__dict__[k]

  def __eq__(self, obj):
    return type(self) == type(obj) and self.__dict__ == obj.__dict__

  def __getitem__(self, k):
    return self.__dict__[k]

  def __getstate__(self):
    return self.__dict__

  def __iter__(self):
    return iter(self.__dict__)

  def __len__(self):
    return len(self.__dict__)

  def __setitem__(self, k, v):
    self.__dict__[k] = v

  def __setattr__(self, k, v):
    if k not in self.__dict__:
      raise AttributeError("'%s' object has no attribute '%s'" % (
          type(self).__name__, k))
    self.__dict__[k] = v

  def __setstate__(self, state):
    self.__init__(state)

  def __repr__(self):
    return "%s(%s)" % (type(self).__name__, pprint.pformat(self.__dict__))

  __str__ = __repr__
