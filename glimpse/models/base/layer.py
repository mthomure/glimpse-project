# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import Image
import numpy as np
from pprint import pformat

from glimpse.backends import ACTIVATION_DTYPE
from glimpse.util.garray import fromimage, toimage
from glimpse.util.dataflow import Node
from glimpse.util.dataflow import node_builder as layer_builder

__all__ = [
    'LayerSpec',
    'Layer',
    'FromImage',
    'ToImage',
    'layer_builder',
    ]

class LayerSpec(Node):
  """Describes a single layer in a model."""

  def __init__(self, id_, name, depends = None):
    super(LayerSpec, self).__init__(id_, depends)
    self.name = name  #: (str) Name of model layer.

  def __repr__(self):
    return "%s(%s)" % (type(self).__name__, ", ".join("%s=%s" % (k,
        pformat(getattr(self, k))) for k in ('id_', 'name', 'depends')))

class Layer(object):
  """Enumerator for model layers."""

  @classmethod
  def FromId(cls, id_):
    """Lookup a LayerSpec object by ID.

    :param id_: Model-unique layer id_ifier.
    :rtype: :class:`LayerSpec`

    Example:

    >>> lyr = Layer.FromId(Layer.IMAGE.id_)
    >>> assert(lyr == Layer.IMAGE)

    """
    if isinstance(id_, LayerSpec):
      return id_
    for layer in cls.AllLayers():
      if layer.id_ == id_:
        return layer
    raise ValueError("Unknown layer id: %r" % id_)

  @classmethod
  def FromName(cls, name):
    """Lookup a LayerSpec object by name.

    This method is not case sensitive.

    :param str name: Layer name.
    :rtype: :class:`LayerSpec`

    Example:

    >>> lyr = Layer.FromName(Layer.IMAGE.name)
    >>> assert(lyr == Layer.IMAGE)

    """
    if isinstance(name, LayerSpec):
      return name
    # Here we perform a linear search, rather than use a dictionary, since we
    # want a case-insensitive comparison. This should be fine, since this method
    # is not called often.
    name_ = name.lower()
    for layer in cls.AllLayers():
      if layer.name.lower() == name_:
        return layer
    raise ValueError("Unknown layer name: %s" % name)

  @classmethod
  def AllLayers(cls):
    """Return the unordered set of all layers.

    :rtype: list of :class:`LayerSpec`

    Example:

    >>> assert(Layer.IMAGE in Layer.AllLayers())

    """
    names = [ k for k in dir(cls) if not k.startswith('_') ]
    values = [ getattr(cls, k) for k in names ]
    return [ v for v in values if isinstance(v, LayerSpec) ]

  @classmethod
  def IsSublayer(cls, sub_layer, super_layer):
    """Determine if one layer appears later in the network than another.

    :param sub_layer: Lower layer.
    :type sub_layer: :class:`LayerSpec`
    :param super_layer: Higher layer.
    :type super_layer: :class:`LayerSpec`
    :rtype: bool

    Examples:

    >>> assert(Layer.IsSublayer(Layer.SOURCE, Layer.IMAGE))
    >>> assert(not Layer.IsSublayer(Layer.IMAGE, Layer.SOURCE))

    """
    for lyr in (super_layer.depends or ()):
      if lyr == sub_layer or cls.IsSublayer(sub_layer, lyr):
        return True
    return False

  @classmethod
  def TopLayer(cls):
    """Determine the top layer in this network.

    The top-most layer is defined as the layer on which no other layer depends.
    If multiple layers meet this criteria, then the first such layer (as
    returned by :meth:`AllLayers`) is returned.

    :rtype: :class:`LayerSpec`

    Example:

    >>> assert(Layer.TopLayer() == Layer.IMAGE)

    """
    all_layers = cls.AllLayers()
    for layer in all_layers:
      if any(layer in (l.depends or ()) for l in all_layers):
        continue
      return layer
    raise Exception("Internal error: Failed to find top layer in network")

def FromImage(input_, backend):
  """Create the initial image layer from some input.

  :param input_: Input data. If array, values should lie in the range [0, 1].
  :type input_: PIL.Image or 2D ndarray of float
  :returns: Image layer data with values in the range [0, 1].
  :rtype: 2D ndarray of float

  """
  if Image.isImageType(input_):
    input_ = fromimage(input_.convert('L'))
    input_ = input_.astype(np.float)
    # Map from [0,255) to [0,1)
    input_ /= 255
    # Note: we could have scaled pixels to [-1,1). This would result in a
    # doubling of the dynamic range of the S1 response, since each S1 activation
    # is of the form:
    #    y = XW
    # where X and W are the input and weight vector, respectively. The rescaled
    # version of X (call it X') is given by:
    #    X' = 2X - 1
    # so the activation is given by
    #    y' = X'W = (2X - 1)W = 2XW - \sum w_i = 2XW
    # since W is a mean-zero Gabor filter. (This ignores retinal processing, and
    # nonlinearities caused by normalization). The scaling of S1 response seems
    # unlikely to cause a significant change in the network output.
  elif isinstance(input_, np.ndarray):
    if input_.ndim != 2:
      raise ValueError("Image array must be 2D")
  else:
    raise ValueError("Unknown input value of type: %s" % type(input_))
  output = backend.PrepareArray(input_)
  if np.isnan(output).any():
    raise BackendError("Found illegal values in image layer")
  return output

def ToImage(data):
  """Create an image from a 2D array of model activity.

  :param data: Single scale of one layer of model activity, with elements in the
     range [0,1].
  :type data: 2D ndarray of floats
  :rtype: Image
  :returns: Greyscale image of layer activity.

  """
  if not (isinstance(data, np.ndarray) and data.ndim == 2 and \
      data.dtype == ACTIVATION_DTYPE):
    raise ValueError("Invalid image layer data")
  data = data.copy()
  data *= 255
  img = toimage(data.astype(np.uint8))
  return img
