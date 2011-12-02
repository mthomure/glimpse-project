# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# Create a 2-part, HMAX-like hierarchy of S+C layers.
# This module implements the "Viz2" model used for the GCNC 2011 experiments.

import copy
import logging
from glimpse.models.misc import LayerSpec, InputSource, SampleC1Patches, \
    DependencyError, AbstractNetwork
from glimpse import util
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

class Model(ModelOps, AbstractNetwork):
  """The Viz2 model. This class adds support for computing an arbitrary layer
  from a given initial model state."""

  # The datatype associated with network states for this model.
  State = State

  # The datatype associated with layer descriptors for this model.
  Layer = Layer

  def BuildSingleNode(self, output_id, state):
    L = self.Layer
    if output_id == L.SOURCE.id:
      raise DependencyError
    elif output_id == L.IMAGE.id:
      return self.BuildImageFromInput(state[L.SOURCE.id].CreateImage())
    elif output_id == L.RETINA.id:
      return self.BuildRetinaFromImage(state[L.IMAGE.id])
    elif output_id == L.S1.id:
      return self.BuildS1FromRetina(state[L.RETINA.id])
    elif output_id == L.C1.id:
      return self.BuildC1FromS1(state[L.S1.id])
    elif output_id == L.S2.id:
      return self.BuildS2FromC1(state[L.C1.id])
    elif output_id == L.C2.id:
      return self.BuildC2FromS2(state[L.S2.id])
    elif output_id == L.IT.id:
      return self.BuildItFromC2(state[L.C2.id])
    raise ValueError("Unknown layer ID: %r" % output_id)

  def GetDependencies(self, output_id):
    return [ l.id for l in self.Layer.FromId(output_id).depends ]

  def BuildLayer(self, output_layer, state, save_all = True):
    """Apply the model through the given layer. This is just a helper for
    AbstractNetwork.BuildNode().
    output_layer -- (Layer or id) output layer to compute
    state -- (State) initial model state from which to compute the output layer
    save_all -- (bool) whether the resulting state should contain values for all
                computed layers in the network, or just the output layer.
    """
    state = copy.copy(state)  # get a shallow copy of the model state
    if isinstance(output_layer, LayerSpec):
      output_layer = output_layer.id
    state = self.BuildNode(output_layer, state)
    if not save_all:
      logging.info("BuildLayer: sparsifying result")
      state_ = State()
      # Keep output layer data
      state_[output_layer] = state[output_layer]
      # Keep source information
      if self.Layer.SOURCE.id in state:
        state_[self.Layer.SOURCE.id] = state[self.Layer.SOURCE.id]
      state = state_
    return state

  def BuildLayerCallback(self, output_layer, save_all = True):
    if isinstance(output_layer, LayerSpec):
      output_layer = output_layer.id
    return LayerBuilder(self, output_layer, save_all)

  def SampleC1Patches(self, num_patches, state, normalize = False):
    """Compute C1 activity and sample a patches from random locations and
    scales.
    num_patches -- (int) number of patches to extract
    normalize -- (bool) whether to normalize each C1 patch
    """
    state = self.BuildLayer(self.Layer.C1, state)
    c1s = state[self.Layer.C1.id]
    patch_it = SampleC1Patches(c1s, kwidth = self.params.s2_kwidth)
    patches = list(itertools.islice(patch_it, num_patches))
    if normalize:
      for patch, location in patches:
        patch /= np.linalg.norm(patch)
    return patches

  def SampleC1PatchesCallback(self, num_patches, normalize = False):
    return C1PatchSampler(self, num_patches, normalize)

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

# Add (circular) Model reference to State class.
State.Model = Model

class LayerBuilder(object):
  """Represents a serializable function that computes a network state
  transformation, computing the value of some network layer and any required
  dependencies."""

  def __init__(self, model, output_layer_id, save_all = True):
    """Create a new object.
    model -- (Model) the model to use when computing layer activity
    output_layer_id -- (scalar) highest layer in model to compute
    save_all -- (bool) whether the resulting state should contain values for all
                computed layers in the network, or just the output layer.
    """
    self.model, self.output_layer_id, self.save_all = model, output_layer_id, \
        save_all

  def __call__(self, state):
    """Transform between network states."""
    return self.model.BuildLayer(self.output_layer_id, state, self.save_all)

  def __str__(self):
    return "%s(model=%s, output_layer=%s, save_all=%s)" % (util.TypeName(self),
        type(self.model), self.output_layer, self.save_all)

  def __repr__(self):
    return "%s(model=%s, output_layer=%s, save_all=%s)" % (util.TypeName(self),
        self.model, self.output_layer, self.save_all)

class C1PatchSampler(object):
  """Represents a serializable function that computes a network state
  transformation for feedforward S->C layer networks. This function computes C1
  activity and then extracts patches from randomly-sampled locations and
  scales."""

  def __init__(self, model, num_patches, normalize = False):
    """Create new object.
    model -- (Model) Viz2 model instantiation to use when computing C1 activity
    num_patches -- (int) number of patches to extract
    normalize -- (bool) whether to normalize each C1 patch
    """
    self.model, self.num_patches, self.normalize = model, num_patches, normalize

  def __call__(self, state):
    """Transform an input model state to a set of C1 patches.
    state -- (State) input model state
    RETURN list of (patch, sample location) pairs
    """
    return self.model.SampleC1Patches(self.num_patches, state, self.normalize)
