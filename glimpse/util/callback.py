"""Defines a function wrapper supporting partial application and serialization.
"""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from pprint import pformat
from cPickle import PicklingError
import types

class Callback(object):
  """A serializable function or instance method."""

  #: Workaround for IPython < 0.13.
  __name__ = "Callback"

  def __init__(self, f, *args, **kw):
    """Create a new Callback object.

    :param callable f: Callable to serialize. This can be a builtin or module-
       level function, a class method, or a bound instance method. This must not
       be a static method, a lambda expression, or an unbound instance method.
    :param args: Positional arguments for invocation.
    :param kw: Keyword arguments for invocation.

    The parameter f can be a builtin, module-level

    """
    self.f = f
    self.args = args
    self.kw = kw

  def __eq__(self, other):
    return type(self) == type(other) and self.__dict__ == other.__dict__

  def __repr__(self):
    values = [pformat(self.f)]
    args_str = ", ".join(map(pformat, self.args))
    if args_str:
      values.append(args_str)
    kw_str = ", ".join("%s=%s" % (pformat(k), pformat(self.kw[k]))
        for k in self.kw)
    if kw_str:
      values.append(kw_str)
    return "%s(%s)" % (type(self).__name__, ", ".join(values))

  def __call__(self, *args, **kw):
    """Call the wrapped function.

    :param args: Additional positional arguments for invocation. Appended to the
       end of any positional arguments specified in the constructor.
    :param kw: Additional keyword arguments for invocation.

    """
    args = self.args + args
    kw_ = dict(self.kw)
    kw_.update(kw)
    return self.f(*args, **kw_)

  def __getstate__(self):
    """Prepare object data for serialization.

    This converts any instance methods to the bound object (if any) and the
    method name.

    """
    f = self.f
    if getattr(f, 'func_name', None) == '<lambda>':
      raise PicklingError("Lambda functions can not be serialized")
    isbound = isinstance(f, types.MethodType)
    if isbound:
      if f.__self__ is None:
        raise PicklingError("Unbound methods can not be serialized")
      f = (f.__self__, f.__func__.__name__)
    return (isbound, f, self.args, self.kw)

  def __setstate__(self, state):
    isbound, self.f = state[:2]
    if isbound:
      # lookup method by name
      self.f = getattr(self.f[0], self.f[1])
    self.args, self.kw = state[2:4]
