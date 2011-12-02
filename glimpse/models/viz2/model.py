# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# Create a 2-part, HMAX-like hierarchy of S+C layers.
# This module implements the "Viz2" model used for the GCNC 2011 experiments.

import copy
import logging
from glimpse.models.misc import LayerSpec, SampleC1Patches, InputSource
import itertools
import numpy as np
from ops import ModelOps

class Layer(object):

  SOURCE = LayerSpec("s", "source")
  IMAGE = LayerSpec("i", "image", SOURCE)
  RETINA = LayerSpec("r", "retina", IMAGE)
  S1 = LayerSpec("s1", "S1", RETINA)
  C1 = LayerSpec("c1", "C1", S1)
  S2 = LayerSpec("s2", "S2", C1)
  C2 = LayerSpec("c2", "C2", S2)
  IT = LayerSpec("it", "IT", C2)

  __LAYERS = { SOURCE.id : SOURCE, IMAGE.id : IMAGE, RETINA.id : RETINA,
      S1.id : S1, C1.id : C1, S2.id : S2, C2.id : C2, IT.id : IT }

  @staticmethod
  def FromId(id):
    layer = Layer.__LAYERS.get(id, None)
    if layer == None:
      raise ValueError("Unknown layer id: %r" % id)
    return layer

  @staticmethod
  def FromName(name):
    # Here we perform a linear search, rather than use a dictionary, since we
    # want a case-insensitive comparison. This should be fine, since this method
    # is not called often.
    name_ = name.lower()
    for layer in Layer.AllLayers():
      if layer.name.lower() == name_:
        return layer
    raise ValueError("Unknown layer name: %s" % name)

  @staticmethod
  def AllLayers():
    """Return the unordered set of all layers."""
    return Layer.__LAYERS.values()

class State(dict):
  """A container for the model state. The main purpose of extending the
  dictionary (instead of just storing states as dictionary objects) is to
  indicate with which model a given state is associated. Similarly, each model
  has a seperate State object, so it is always clear which model generated a
  given state object."""

  def __str__(self):
    return "%s(%s)" % (util.TypeName(self),
        ", ".join(map(str, self.keys())))

  def __repr__(self):
    reps = dict((k, repr(v)) for k, v in self.items())
    if len(reps.keys()) > 0:
      data = "%s\n" % "".join("\n  %s = %s" % \
          (Layer.FromId(k).name, reps[k]) for k in reps.keys())
    else:
      data = ""
    return "%s(%s)" % (util.TypeName(self), data)

  def __eq__(self, other):
    if not isinstance(other, type(self)):
      return False
    if set(self.keys()) != set(other.keys()):
      return False
    for k in self:
      a, b = self[k], other[k]
      if isinstance(a, np.ndarray):
        # Compare single-scale layers
        return np.all(a, b)
      # Compare multi-scale layers
      return np.all(a_ == b_ for a_, b_ in zip(a, b))

class Model(ModelOps):
  """The Viz2 model. This class adds support for computing an arbitrary layer
  from a given initial model state."""

  def MakeStateFromFilename(self, filename):
    """Create a model state with a single SOURCE layer.
    filename -- (str) path to an image file
    RETURN (State) the new model state
    """
    state = self.State()
    state[self.Layer.SOURCE.id] = InputSource(filename)
    return state

  def MakeStateFromImage(self, image):
    """Create a model state with a single IMAGE layer.
    image -- Image or (2-D) array of input data. If array, values should lie in
             the range [0, 1].
    RETURN (State) the new model state
    """
    state = self.State()
    state[self.Layer.IMAGE.id] = self.BuildImageFromInput(image)
    return state

  def _ComputeLayerActivity(self, layer, input_layers):
    # Note that the input layers are indexed by ID, not layer object. This is
    # done to allow a copied layer object to be passed to BuildLayer().
    if layer == Layer.RETINA:
      return self.BuildRetinaFromImage(input_layers[Layer.IMAGE.id])
    elif layer == Layer.S1:
      return self.BuildS1FromRetina(input_layers[Layer.RETINA.id])
    elif layer == Layer.C1:
      return self.BuildC1FromS1(input_layers[Layer.S1.id])
    elif layer == Layer.S2:
      return self.BuildS2FromC1(input_layers[Layer.C1.id])
    elif layer == Layer.C2:
      return self.BuildC2FromS2(input_layers[Layer.S2.id])
    elif layer == Layer.IT:
      return self.BuildItFromC2(input_layers[Layer.C2.id])
    raise ValueError("Can't compute activity for layer: %s" % layer.name)

  def _BuildLayerHelper(self, layer, state):
    """Recursively fulfills dependencies by building parent layers."""
    # Short-circuit computation if data exists
    if layer.id in state and state[layer.id] != None:
      return state
    # Handle image layer as special case
    if layer == Layer.IMAGE:
      if Layer.SOURCE.id not in state:
        raise DependencyException("Can't build image layer without input "
            "source.")
      img = state[Layer.SOURCE.id].CreateImage()
      state[Layer.IMAGE.id] = self.BuildImageFromInput(img)
    # Handle all layers above the image layer.
    else:
      # Compute any dependencies
      for parent_layer in layer.depends:
        self._BuildLayerHelper(parent_layer, state)
      input_layers = dict((x.id, state[x.id]) for x in layer.depends)
      # Compute requested layer
      state[layer.id] = self._ComputeLayerActivity(layer, input_layers)

  def BuildLayer(self, input_state, layer):
    """Apply the model through the given layer.
    input_state -- (State) initial data for the model (e.g., an image, or
                   activity for an intermediate layer).
    layer -- (Layer) identifier of output layer to compute
    """
    # Perform shallow copy of existing state
    output_state = copy.copy(input_state)
    # Recursively compute activity up through the given layer
    self._BuildLayerHelper(layer, output_state)
    return output_state

  # Links to model-specific implementations of common functions and classes.
  def C1PatchSampler(self, *args, **kwargs):
    return C1PatchSampler(self, *args, **kwargs)

  def ModelTransform(self, *args, **kwargs):
    return ModelTransform(self, *args, **kwargs)

  Layer = Layer

#### FUNCTION OBJECTS ####

class C1PatchSampler(object):
  """Represents a model state transformation through C1, which then extracts
  patches from randomly-sampled locations and scales."""

  def __init__(self, model, samples_per_image, normalize = False):
    """Create new object.
    model -- (Model) Viz2 model instantiation to use when computing C1 activity
    samples_per_image -- (int) number of patches to extract from each image
    """
    self.model, self.samples_per_image, self.normalize = model, \
        samples_per_image, normalize

  def __call__(self, state):
    """Transform an input model state to an output state.
    RETURN list of (prototype, sample location) pairs
    """
    state = self.model.BuildLayer(state, Layer.C1)
    c1s = state[Layer.C1.id]
    proto_it = SampleC1Patches(c1s, kwidth = self.model._params.s2_kwidth)
    protos = list(itertools.islice(proto_it, self.samples_per_image))
    if self.normalize:
      for proto, location in protos:
        proto /= np.linalg.norm(proto)
    return protos

class ModelTransform(object):
  """Represents a model state transformation that computes through some model
  layer."""

  def __init__(self, model, layer, save_all = True):
    """Create a new object.
    model -- the model to use when computing layer activity
    layer -- highest layer in model to compute
    save_all -- (bool) whether transform should save all computed layers in the
                result, or just the output layer.
    """
    self.model, self.layer, self.save_all = model, layer, save_all

  def __call__(self, state):
    """Transform between model states."""
    state = self.model.BuildLayer(state, self.layer)
    if not self.save_all:
      logging.info("ModelTransform: sparsifying result")
      for layer_id in state.keys():
        if layer_id != self.layer.id:
          del state[layer_id]
    return state

  def __str__(self):
    return "ModelTransform(model=%s, layer=%s, save_all=%s)" % (self.model,
        self.layer, self.save_all)

  __repr__ = __str__
