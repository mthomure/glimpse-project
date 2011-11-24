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
  IMAGE = LayerSpec(0, "image")
  RETINA = LayerSpec(1, "retina", IMAGE)
  S1 = LayerSpec(2, "S1", RETINA)
  C1 = LayerSpec(3, "C1", S1)
  S2 = LayerSpec(4, "S2", C1)
  C2 = LayerSpec(5, "C2", S2)
  IT = LayerSpec(6, "IT", C2)

  __LAYERS = (IMAGE, RETINA, S1, C1, S2, C2, IT)
  @staticmethod
  def FromId(id):
    if id < 0 or id >= len(Layer.__LAYERS):
      raise ValueError("Unknown layer id: %s" % id)
    return Layer.__LAYERS[id]

  @staticmethod
  def FromName(name):
    names = name.lower()
    for layer in Layer.__LAYERS:
      if layer.name.lower() == name:
        return layer
    raise ValueError("Unknown layer name: %s" % name)

  @staticmethod
  def Layers():
    return Layer.__LAYERS

class LayerData(object):
  """Container for data of layer that only generates unit activities."""

  # Activity for all units in this layer. This is either a single 2-D array (in
  # the case of IMAGE and RETINA layers), or a list of N-D arrays (in all other
  # cases).
  activity = None

  def __init__(self, activity = None):
    self.activity = activity

  def __repr__(self):
    return "LayerData(%r)" % (self.activity,)

  def __eq__(self, other):
    if not isinstance(other, LayerData):
      return False
    # Compare single-scale layers
    if isinstance(self.activity, np.ndarray):
      return np.all(self.activity == other.activity)
    # Compare multi-scale layers
    return np.all(s == o for s, o in zip(self.activity, other.activity))

class State(dict):
  """Stores data for all Viz2 model layers. All elements should be instances of
  LayerData. Layer output is not stored directly in the state object, so that
  it is clear when a layer has been computed. Data is indexed by layer ID."""

  # The input. Should be an instance of InputSource.
  source = None

  def __init__(self, source = None):
    self.source = source

  def __str__(self):
    return "State(source=%s, data=[%s])" % (self.source,
        ", ".join(map(str, self.keys())))

  def __repr__(self):
    reps = {}
    for k in self.keys():
      if k in (Layer.IMAGE.id, Layer.RETINA.id):
        # Data for image and retinal layers is single 2-D map.
        reps[k] = self[k].activity.shape
      else:
        # Data for other layers is sequence of N-D maps.
        reps[k] = ", ".join(str(scale.shape) for scale in self[k].activity)
    if len(reps.keys()) > 0:
      data = "{%s\n}" % "".join("\n  %s = %s(%s)" % \
          (k, Layer.FromId(k).name, reps[k]) for k in reps.keys())
    else:
      data = "no data"
    return "State(source=%s, %s)" % (self.source, data)

  def __eq__(self, other):
    if not (isinstance(other, State) and self.source == other.source):
      return False
    if set(self.keys()) != set(other.keys()):
      return False
    return np.all([ self[k] == other[k] for k in self ])

class Model(ModelOps):
  """The Viz2 model. This class adds support for computing an arbitrary layer
  from a given initial model state."""

  def GetLayers(self):
    return (Layer.IMAGE, Layer.RETINA, Layer.S1, Layer.C1, Layer.S2, Layer.C2,
        Layer.IT)

  def MakeStateFromFilename(self, filename):
    return State(InputSource(filename))

  def MakeStateFromImage(self, image):
    state = State()
    state[Layer.IMAGE.id] = LayerData(self.BuildImageFromInput(image))
    return state

  def _ComputeLayerActivity(self, layer, input_layers):
    # Note that the input layers are indexed by ID, not layer object. This is
    # done to allow a copied layer object to be passed to BuildLayer().
    if layer == Layer.RETINA:
      return LayerData(self.BuildRetinaFromImage(
          input_layers[Layer.IMAGE.id].activity))
    elif layer == Layer.S1:
      return LayerData(self.BuildS1FromRetina(
          input_layers[Layer.RETINA.id].activity))
    elif layer == Layer.C1:
      return LayerData(self.BuildC1FromS1(input_layers[Layer.S1.id].activity))
    elif layer == Layer.S2:
      return LayerData(self.BuildS2FromC1(input_layers[Layer.C1.id].activity))
    elif layer == Layer.C2:
      return LayerData(self.BuildC2FromS2(input_layers[Layer.S2.id].activity))
    elif layer == Layer.IT:
      return LayerData(self.BuildItFromC2(input_layers[Layer.C2.id].activity))
    raise ValueError("Can't compute activity for layer: %s" % layer.name)

  def _BuildLayerHelper(self, layer, state):
    """Recursively fulfills dependencies by building parent layers."""
    # Short-circuit computation if data exists
    if layer.id in state and state[layer.id] != None:
      return state
    # Handle image layer as special case
    if layer == Layer.IMAGE:
      if state.source == None:
        raise DependencyException("Can't build image layer without input "
            "source.")
      img = state.source.CreateImage()
      state[Layer.IMAGE.id] = LayerData(self.BuildImageFromInput(img))
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

#### EXECUTOR FUNCTIONS ####

class C1PatchSampler(object):
  """Represents a model state transformation through C1, which then extracts
  patches from randomly-sampled locations and scales."""

  def __init__(self, model, num_patches, normalize = False):
    """Create new object.
    model -- (Model) Viz2 model instantiation to use when computing C1 activity
    num_patches -- (int) number of patches to extract
    """
    self.model, self.num_patches, self.normalize = model, num_patches, normalize

  def __call__(self, state):
    """Transform between model states."""
    state = self.model.BuildLayer(state, Layer.C1)
    c1s = state[Layer.C1.id].activity
    proto_it = SampleC1Patches(c1s, kwidth = self.model._params.s2_kwidth)
    protos = list(itertools.islice(proto_it, self.num_patches))
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
