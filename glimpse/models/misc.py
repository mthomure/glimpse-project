"""General functions that are applicable to multiple models."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import os
import importlib

__all__ = [
    'GetModelClass',
    'MakeParams',
    'MakeModel',
    'DEFAULT_MODEL_NAME',
]

# NOTE: All model sub-packages should export the symbols 'Model', 'Layer', and
# 'Params'.

#: Model name used by :func:`GetModelClass`, :func:`MakeModel`, and
#: :func:`MakeParams` when no name is supplied. This should not be "base".
DEFAULT_MODEL_NAME = "ml"

def GetModelClass(name = None):
  """Lookup a Glimpse model class by name.

  :param str name: The name of the model. This corresponds to the model's
     package name. The default is read from the :envvar:`GLIMPSE_MODEL`
     environment variable, or is `DEFAULT_MODEL_NAME` if this is not set.

  Examples:

  Create a new instance of the 'ml' model:

  >>> ModelClass = GetModelClass("ml")
  >>> assert(ModelClass == glimpse.ml.Model)

  """
  if name == None:
    name = os.environ.get('GLIMPSE_MODEL', DEFAULT_MODEL_NAME)
  pkg = importlib.import_module(".%s" % name, __package__)
  try:
    return getattr(pkg, 'Model')
  except AttributeError:
    raise ValueError("Unknown model name: %s" % name)

def MakeParams(name = None, **kw):
  """Create parameters for the given model."""
  return GetModelClass(name).ParamClass(**kw)

def MakeModel(*args):
  """Create an instance of the given model.

  Usage:

  model = MakeModel(name)
  model = MakeModel(params)
  model = MakeModel(name, params)

  :param str name: The name of the model.
  :param params: Parameters for the given model.
  :returns: Created model instance.

  """
  name = params = None
  if len(args) == 1:
    if isinstance(args[0], basestring):
      name = args[0]
    else:
      name = None
      params = args[0]
  elif len(args) == 2:
    name, params = args[:2]
  elif len(args) > 2:
    raise ValueError("Too many arguments")
  return GetModelClass(name)(params = params)
