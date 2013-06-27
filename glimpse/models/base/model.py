# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import copy as copy_mod
import Image
import numpy as np

from glimpse.util.dataflow import DataFlow, State as BaseState
from glimpse.backends import (MakeBackend, InputSource, BackendError,
    ACTIVATION_DTYPE)

from .misc import *
from .param import Params
from . import layer
from .layer import *

__all__ = [
    'Layer',
    'State',
    'Model',
    'layer_builder',
    ]

class Layer(layer.Layer):

  #: Specifier for the source of the image data. This may be :obj:`None` if the
  #: data was generated programatically, or an :class:`InputSource` object if
  #: the data came from an image file.
  SOURCE = LayerSpec("s", "source")

  #: Specifier for the raw input data.
  IMAGE = LayerSpec("i", "image", [SOURCE])

class State(BaseState):
  """A dictionary container for the :class:`Model` state.

  The main purpose of extending the dictionary (instead of just storing states
  as dictionary objects) is to indicate with which model a given state is
  associated. Each model has a seperate state object, so it is always clear
  which model generated a given state object.

  """

  #: The datatype associated with the model for this state. Should be set by
  #: sub-class.
  ModelClass = None

class Model(DataFlow):
  """Abstract base class for a Glimpse model."""

  #: The datatype associated with layer descriptors for this model. This should
  #: be over-ridden by the sub-class with a descendent of :class:`layer.Layer`.
  LayerClass = Layer

  #: The type of the parameter collection associated with this model. This
  #: should be over-ridden by the sub-class.
  ParamClass = Params

  #: The datatype associated with network states for this model. This should be
  #: over-ridden by the sub-class, and should generally be a descendent of
  #: :class:`state.State`.
  StateClass = State

  def __init__(self, backend = None, params = None):
    """Create new object.

    This method can als be called as __init__(backend) or __init__(params).

    :param backend: Implementation of backend operations, such as dot-products.
    :param params: Model configuration.

    """
    super(Model, self).__init__()
    if params == None and isinstance(backend, Params):
      params = backend
      backend = None
    if backend == None:
      backend = MakeBackend()
    if params == None:
      params = self.ParamClass()
    else:
      if not isinstance(params, self.ParamClass):
        raise ValueError("Params object has wrong type: expected %s, got %s" % \
            (self.ParamClass, type(params)))
    self.backend = backend
    self.params = params

  def MakeState(self, source, copy = False):
    """Create a model state wrapper for the given image source.

    :type state: str or Image.Image or 2D array of ACTIVATION_DTYPE or `State`
       subclass
    :param state: Source information
    :param bool copy: If the input is already a state object, this argument
       determines whether the state is copied.
    :rtype: `state.State` subclass

    If `source` is an array, values should lie in the range [0,1).

    """
    if isinstance(source, self.StateClass):
      if copy:
        source = copy_mod.copy(source)  # shallow copy
      return source
    state = self.StateClass()
    if isinstance(source, basestring):
      state[self.LayerClass.SOURCE] = InputSource(source)
    elif Image.isImageType(source):
      img = PrepareImage(source, self.params)
      state[self.LayerClass.IMAGE] = layer.FromImage(img, self.backend)
    elif isinstance(source, np.ndarray):
      if source.ndim != 2:
        raise ValueError("Array inputs must be 2D")
      if source.dtype != ACTIVATION_DTYPE:
        raise ValueError("Array values must have type: %s" % ACTIVATION_DTYPE)
      state[self.LayerClass.IMAGE] = source
    else:
      raise ValueError("Image source had unknown type: %s" % source)
    return state

  @layer_builder(Layer.IMAGE)
  def BuildImageLayer(self, source):
    img = source.CreateImage()
    img = PrepareImage(img, self.params)
    return layer.FromImage(img, self.backend)

State.ModelClass = Model
