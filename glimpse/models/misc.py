# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# General functions that are applicable to multiple models.

from glimpse.util import ImageToArray
import Image
import numpy as np
import random

class LayerSpec(object):
  """Describes a single layer in a model."""

  # A unique (within a given model) identifier for the layer. Not necessarily
  # numeric.
  id = 0

  # A user-friendly name for the layer.
  name = ""

  # Parents whose data is required to compute this layer.
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

  def __init__(self, source = None):
    super(Exception, self).__init__()
    self.source = source

class InputSource(object):
  """Describes the input to a hierarchical model. Examples include the path to a
  single image, or the path and frame of a video."""

  # Path to an image file.
  image_path = None

  def __init__(self, image_path = None):
    if image_path != None and not isinstance(image_path, basestring):
      raise ValueError("Image path must be a string")
    self.image_path = image_path

  def CreateImage(self):
    """Create a new PIL.Image object for this input source."""
    try:
      return Image.open(self.image_path)
    except IOError:
      raise InputSourceLoadException(self)

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
  input_ -- Image or (2-D) array of input data. If array, values should lie in
            the range [0, 1].
  RETURNS (2-D) array containing image layer data
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
  c1s - (4-D array, or list of 3-D arrays) C1 activity maps, one map per scale.
  RETURNS: infinite iterator over prototype arrays and corresponding locations
           (i.e., iterator elements are 2-tuples). Prototype location gives
           top-left corner of C1 region.
  """
  assert (np.array(len(c1s[0].shape)) == 3).all()
  num_scales = len(c1s)
  num_orientations = len(c1s[0])
  while True:
    scale = random.randint(0, num_scales - 1)
    c1 = c1s[scale]
    height, width = c1.shape[-2:]
    y = random.randint(0, height - kwidth)
    x = random.randint(0, width - kwidth)
    patch = c1[ :, y : y + kwidth, x : x + kwidth ]
    yield patch.copy(), (scale, y, x)

class DependencyError(Exception):
  """Indicates that a dependency required to build a node in the network is
  unavailable."""

class AbstractNetwork(object):

  def BuildSingleNode(self, output_id, state):
    """Constructs a given node from one or more pre-computed input nodes.
    output_id -- unique (at least in this network) identifier for output node
    input_values -- (dict) set of node data for inputs (will not be modified)
    RETURN scalar output value for node
    """
    raise NotImplemented

  def GetDependencies(self, output_id):
    """Encodes the dependency structure for each node in the network, providing
    identifiers for the set of inputs required to build a given output node.
    output_id -- unique identifier for output node
    RETURN (list) identifiers of required input nodes
    """
    raise NotImplemented

  def BuildNode(self, output_id, state):
    """Construct the given output value, recursively building required
    dependencies. A node is considered to be "built" if the node's key is set in
    the state dictionary, even if the corresponding value is None.
    output_id -- unique identifier for the node to compute
    state -- (dict) current state of the network, which may be used to store the
             updated network state.
    RETURN (dict) the final state of the network, which is guaranteed to
           contain a value for the output node.
    """
    # Short-circuit computation if data exists
    if output_id not in state:
      # Compute any dependencies
      for node in self.GetDependencies(output_id):
        state = self.BuildNode(node, state)
      # Compute the output node
      state[output_id] = self.BuildSingleNode(output_id, state)
    return state
