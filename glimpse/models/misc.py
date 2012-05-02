# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# General functions that are applicable to multiple models.

from glimpse.util import ImageToArray
from glimpse.backends import InsufficientSizeException
import Image
import logging
import numpy as np
import random

class LayerSpec(object):
  """Describes a single layer in a model."""

  #: A unique (within a given model) identifier for the layer. Not necessarily
  #: numeric.
  id = 0

  #: A user-friendly name for the layer.
  name = ""

  #: Parents whose data is required to compute this layer.
  depends = []

  def __init__(self, id, name = None, *depends):
    self.id, self.name, self.depends = id, name, depends

  def __str__(self):
    return self.name

  def __repr__(self):
    return "LayerSpec(id=%s, name=%s, depends=[%s])" % (self.id, self.name,
        ", ".join(map(str, self.depends)))

  def __eq__(self, other):
    # XXX this is only useful for comparing layers for a single model (since it
    # only uses the id attribute).
    return isinstance(other, LayerSpec) and self.id == other.id

class InputSourceLoadException(Exception):
  """Thrown when an input source can not be loaded."""

  def __init__(self, msg = None, source = None):
    super(Exception, self).__init__(msg)
    self.source = source

  def __str__(self):
    return "InputSourceLoadException(%s)" % self.source

  __repr__ = __str__

class InputSource(object):
  """Describes the input to a hierarchical model.

  Examples include the path to a single image, or the path and frame of a video.

  """

  #: Path to an image file.
  image_path = None

  #: Size of minimum dimension when resizing
  resize = None

  def __init__(self, image_path = None, resize = None):
    if image_path != None and not isinstance(image_path, basestring):
      raise ValueError("Image path must be a string")
    if resize != None:
      resize = int(resize)
      if resize <= 0:
        raise ValueError("Resize value must be positive")
    self.image_path = image_path
    self.resize = resize

  def CreateImage(self):
    """Reads image from this input source.

    :rtype: PIL.Image

    """
    try:
      img = Image.open(self.image_path)
    except IOError:
      raise InputSourceLoadException("I/O error while loading image", source = self)
    if self.resize != None:
      w, h = img.size
      ratio = float(self.resize) / min(w, h)
      new_size = int(w * ratio), int(h * ratio)
      logging.info("Resize image %s from (%d, %d) to %s" % (self.image_path, w,
          h, new_size))
      img = img.resize(new_size, Image.BICUBIC)
    return img

  def __str__(self):
    return self.image_path

  def __repr__(self):
    return "InputSource(image_path=%s)" % self.image_path

  def __eq__(self, other):
    return isinstance(other, InputSource) and \
        self.image_path == other.image_path

  def __ne__(self, other):
    return not (self == other)

def ImageLayerFromInputArray(input_, backend):
  """Create the initial image layer from some input.

  :param input_: Input data. If array, values should lie in the range [0, 1].
  :type input_: PIL.Image or 2D ndarray of float
  :returns: Image layer data.
  :rtype: 2D ndarray of float

  """
  if isinstance(input_, Image.Image):
    input_ = input_.convert('L')
    input_ = ImageToArray(input_, transpose = True)
    input_ = input_.astype(np.float)
    # Map from [0, 255] to [0, 1]
    input_ /= 255
  return backend.PrepareArray(input_)

def SampleC1Patches(c1s, kwidth):
  """
  Sample patches from a layer of C1 activity.

  :param c1s: C1 activity maps, one map per scale.
  :type c1s: 4D ndarray, or list of 3D ndarray
  :returns: Infinite iterator over prototype arrays and corresponding locations
     (i.e., iterator elements are 2-tuples). Prototype location gives top-left
     corner of C1 region.

  """
  assert (np.array(len(c1s[0].shape)) == 3).all()
  num_scales = len(c1s)
  num_orientations = len(c1s[0])
  while True:
    scale = random.randint(0, num_scales - 1)
    c1 = c1s[scale]
    height, width = c1.shape[-2:]
    if height <= kwidth or width <= kwidth:
      raise InsufficientSizeException()
    y = random.randint(0, height - kwidth)
    x = random.randint(0, width - kwidth)
    patch = c1[ :, y : y + kwidth, x : x + kwidth ]
    yield patch.copy(), (scale, y, x)

class DependencyError(Exception):
  """Indicates that a dependency required to build a node in the network is
  unavailable."""

class AbstractNetwork(object):
  """An abstract base class for a Glimpse model.

  This class provides the scafolding for recursive dependency satisfaction.

  """

  def BuildSingleNode(self, output_id, state):
    """Constructs a given node from one or more pre-computed input nodes.

    :param output_id: Unique (at least in this network) identifier for output
       node.
    :param dict input_values: Node data for inputs (will not be modified).
    :returns: Scalar output value for node.

    """
    raise NotImplemented

  def GetDependencies(self, output_id):
    """Encodes the dependency structure for each node in the network.

    :param output_id: Unique identifier for output node.
    :returns: Identifiers of nodes required to build the given output node.
    :rtype: list

    """
    raise NotImplemented

  def BuildNode(self, output_id, state):
    """Construct the given output value.

    The output value is built by recursively building required dependencies. A
    node is considered to be *built* if the node's key is set in the state
    dictionary, even if the corresponding value is None.

    :param output_id: Unique identifier for the node to compute.
    :param dict state: Current state of the network, which may be used to store
       the updated network state.
    :returns: The final state of the network, which is guaranteed to contain a
       value for the output node.
    :rtype: dict

    """
    # Short-circuit computation if data exists
    if output_id not in state:
      # Compute any dependencies
      for node in self.GetDependencies(output_id):
        state = self.BuildNode(node, state)
      # Compute the output node
      state[output_id] = self.BuildSingleNode(output_id, state)
    return state
