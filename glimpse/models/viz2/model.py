# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import copy
import logging
from glimpse.backends import InsufficientSizeException
from glimpse.models.misc import LayerSpec, InputSource, SampleC1Patches, \
    DependencyError, AbstractNetwork
from glimpse import pools, util
import itertools
from math import sqrt
import numpy as np
from ops import ModelOps
import random

class Layer(object):

  #: Specifier for the source of the image data. This may be :obj:`None` if the
  #: data was generated programatically, or an :class:`InputSource` object if
  #: the data came from an image file.
  SOURCE = LayerSpec("s", "source")

  #: Specifier for the raw input data.
  IMAGE = LayerSpec("i", "image", SOURCE)

  #: Specifier for the preprocessed input.
  RETINA = LayerSpec("r", "retina", IMAGE)

  #: Specifier for the result of S1 filtering.
  S1 = LayerSpec("s1", "S1", RETINA)

  #: Specifier for the result of C1 pooling.
  C1 = LayerSpec("c1", "C1", S1)

  #: Specifier for the result of S2 filtering.
  S2 = LayerSpec("s2", "S2", C1)

  #: Specifier for the result of C2 (local) pooling.
  C2 = LayerSpec("c2", "C2", S2)

  #: Specifier for the result of IT (global) pooling.
  IT = LayerSpec("it", "IT", C2)

  __LAYERS = { SOURCE.id : SOURCE, IMAGE.id : IMAGE, RETINA.id : RETINA,
      S1.id : S1, C1.id : C1, S2.id : S2, C2.id : C2, IT.id : IT }

  @staticmethod
  def FromId(id):
    """Lookup a LayerSpec object by ID."""
    layer = Layer.__LAYERS.get(id, None)
    if layer == None:
      raise ValueError("Unknown layer id: %r" % id)
    return layer

  @staticmethod
  def FromName(name):
    """Lookup a LayerSpec object by name."""
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
  """A dictionary container for the :class:`Model` state.

  The main purpose of extending the dictionary (instead of just storing states
  as dictionary objects) is to indicate with which model a given state is
  associated. Similarly, each model has a seperate state object, so it is always
  clear which model generated a given state object.

  """

  #: The datatype associated with the model for this state.
  ModelClass = None

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
  """Create a 2-part, HMAX-like hierarchy of S+C layers.

  This module implements the "Viz2" model used for the GCNC 2011 experiments.

  """

  #: The datatype associated with network states for this model.
  StateClass = State

  #: The datatype associated with layer descriptors for this model.
  LayerClass = Layer

  def BuildSingleNode(self, output_id, state):
    """Compute activity for a single model layer.

    This method is called by :meth:`BuildNode`.

    :param output_id: Layer to compute.
    :type output_id: :class:`LayerSpec`
    :param state: Model state from which to compute given layer.
    :type state: :class:`State`
    :returns: Activity for given output layer.
    :rtype: ndarray

    """
    L = self.LayerClass
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
    """Find the set of dependencies for a given model layer.

    :rtype: list of :class:`LayerSpec`

    """
    return [ l.id for l in self.LayerClass.FromId(output_id).depends ]

  def BuildLayer(self, output_layer, state, save_all = True):
    """Apply the model through to the given layer.

    :param output_layer: Output layer to compute.
    :type output_layer: :class:`LayerSpec` or scalar (layer ID)
    :param state: Initial model state from which to compute the output layer.
    :type state: :class:`State`
    :param bool save_all: Whether the resulting state should contain values for
       all computed layers in the network, or just the output layer.
    :returns: Output state containing the given layer
    :rtype: :class:`State`

    """
    state = copy.copy(state)  # get a shallow copy of the model state
    if isinstance(output_layer, LayerSpec):
      output_layer = output_layer.id
    try:
      output_state = self.BuildNode(output_layer, state)
    except InsufficientSizeException, e:
      # Try to annotate exception with source information.
      source = state.get(self.LayerClass.SOURCE.id, None)
      if source == None:
        raise
      raise InsufficientSizeException(source = source)
    if not save_all:
      state_ = State()
      # Keep output layer data
      state_[output_layer] = state[output_layer]
      # Keep source information
      if self.LayerClass.SOURCE.id in state:
        state_[self.LayerClass.SOURCE.id] = state[self.LayerClass.SOURCE.id]
      state = state_
    return state

  def BuildLayerCallback(self, output_layer, save_all = True):
    """Create a function that will apply the model through to the given layer.

    :param output_layer: Output layer to compute.
    :type output_layer: :class:`LayerSpec` or scalar (layer id)
    :param state: Initial model state from which to compute the output layer.
    :type state: :class:`State`
    :param bool save_all: Whether the resulting state should contain values for
       all computed layers in the network, or just the output layer.
    :returns: A callable object that, when evaluated, will apply the model.
    :rtype: :class:`LayerBuilder`

    """
    if isinstance(output_layer, LayerSpec):
      output_layer = output_layer.id
    return LayerBuilder(self, output_layer, save_all)

  def SampleC1Patches(self, num_patches, state, normalize = False):
    """Compute C1 activity and sample patches from random locations and scales.

    :param int num_patches: Number of patches to extract.
    :param bool normalize: Whether to normalize each C1 patch.
    :returns: The set of patches
    :rtype: list of (patch, location) pairs

    """
    state = self.BuildLayer(self.LayerClass.C1, state)
    c1s = state[self.LayerClass.C1.id]
    try:
      patch_it = SampleC1Patches(c1s, kwidth = self.params.s2_kwidth)
      patches = list(itertools.islice(patch_it, num_patches))
    except InsufficientSizeException, e:
      # Try to annotate exception with source information.
      source = state.get(self.LayerClass.SOURCE.id, None)
      if source == None:
        raise
      raise InsufficientSizeException(source = source)
    # TEST CASE: single state with uniform C1 activity and using normalize=True,
    # check that result does not contain NaNs.
    if normalize:
      for patch, location in patches:
        n = np.linalg.norm(patch)
        if n == 0:
          logging.warn("Normalizing empty C1 patch")
          patch[:] = 1. / sqrt(patch.size)
        else:
          patch /= n
    return patches

  def SampleC1PatchesCallback(self, num_patches, normalize = False):
    """Create a function that will sample patches from random locations and
    scales.

    :param int num_patches: Number of patches to extract.
    :param bool normalize: Whether to normalize each C1 patch.
    :return: A callable object that, when evaluated, will sample patches
    :rtype: :class:`C1PatchSampler`

    """
    return C1PatchSampler(self, num_patches, normalize)

  def MakeStateFromFilename(self, filename, resize = None):
    """Create a model state with a single SOURCE layer.

    :param str filename: Path to an image file.
    :param int resize: Scale minimum edge to fixed length.
    :returns: The new model state.
    :rtype: :class:`State`

    """
    state = self.StateClass()
    state[self.LayerClass.SOURCE.id] = InputSource(filename, resize = resize)
    return state

  def MakeStateFromImage(self, image):
    """Create a model state with a single IMAGE layer.

    :param image: Input data. If array, values should lie in the range [0, 1].
    :type image: PIL.Image or 2D ndarray
    :param int resize: Scale minimum edge to fixed length.
    :returns: The new model state.
    :rtype: :class:`State`

    """
    state = self.StateClass()
    state[self.LayerClass.IMAGE.id] = self.BuildImageFromInput(image)
    return state

  def ImprintS2Prototypes(self, num_prototypes, input_states, normalize = True,
      pool = None):
    """Imprint a set of S2 prototypes from a set of images.

    :param int num_prototypes: Total number of prototypes to generate.
    :param input_states: initial network states from which to compute C1.
    :type input_states: list of State
    :param bool normalize: Whether each prototype is scaled to have unit norm.
    :param pool: Worker pool used to evaluate the model.
    :returns: A numpy array of prototypes, and a list of prototype locations.

    """
    if pool == None:
      pool = pools.MakePool()
    if num_prototypes < len(input_states):
      # Take a random subset of images.
      random.shuffle(input_states)
      input_states = input_states[:num_prototypes]
      patches_per_image = 1
    else:
      patches_per_image, extra = divmod(num_prototypes, len(input_states))
      if extra > 0:
        patches_per_image += 1
    sampler = self.SampleC1PatchesCallback(patches_per_image,
        normalize = normalize)
    # Compute C1 activity, and sample patches.
    values_per_image = pool.map(sampler, input_states)
    # We now have a iterator over (prototype,location) pairs for each image.
    # Evaluate the iterator, and store the result as a list. Note that we must
    # evaluate before concatenating (below) to ensure we don't leave requests
    # sitting on the worker pool.
    values_per_image = map(list, values_per_image)
    # Add input state index to locations.
    assert len(input_states) == len(values_per_image), \
        "Expected %d C1 patch sets, but got %d" % (len(input_states),
            len(values_per_image))
    for idx in range(len(input_states)):
      values = values_per_image[idx]
      # Locations are tuples of (scale, y, x). Prepend input state index.
      values = [ (p, (idx,) + l) for p, l in values ]
      values_per_image[idx] = values
    # Chain them together into a single iterator.
    all_values = list(itertools.chain(*values_per_image))
    # If the number of requested prototypes is not an even multiple of the
    # number of images, we have imprinted too many prototypes. Crop them.
    all_values = all_values[:num_prototypes]
    # Convert list of tuples to tuple of lists.
    prototypes, locations = zip(*all_values)
    # Convert prototype to a single numpy array.
    prototypes = np.array(prototypes, util.ACTIVATION_DTYPE)
    return prototypes, locations

# Add (circular) Model reference to State class.
State.ModelClass = Model

class LayerBuilder(object):
  """Represents a serializable function.

  This function computes a network state transformation, computing the value of
  some network layer and any required dependencies.

  """

  __name__ = "LayerBuilder"  # needed for use with IPython.parallel

  def __init__(self, model, output_layer_id, save_all = True):
    """Create a new object.

    :param model: The model to use when computing layer activity.
    :type model: :class:`Model`
    :param output_layer_id: Highest layer in model to compute.
    :type output_layer_id: :class:`LayerSpec` or scalar (layer id)
    :param bool save_all: Whether the resulting state should contain values for
       all computed layers in the network, or just the output layer.

    """
    self.model, self.output_layer_id, self.save_all = model, output_layer_id, \
        save_all

  def __call__(self, state):
    """Transform between network states."""
    return self.model.BuildLayer(self.output_layer_id, state, self.save_all)

  def __str__(self):
    return "%s(model=%s, output_layer_id=%s, save_all=%s)" % \
        (util.TypeName(self), type(self.model), self.output_layer_id,
        self.save_all)

  def __repr__(self):
    return "%s(model=%s, output_layer_id=%s, save_all=%s)" % \
        (util.TypeName(self), self.model, self.output_layer_id, self.save_all)

class C1PatchSampler(object):
  """Represents a serializable function.

  This function computes a network state transformation for feedforward S->C
  layer networks. This function computes C1 activity and then extracts patches
  from randomly-sampled locations and
  scales.

  """

  __name__ = "C1PatchSampler"  # needed for use with IPython.parallel

  def __init__(self, model, num_patches, normalize = False):
    """Create new object.

    :param model: Viz2 model instantiation to use when computing C1 activity.
    :type: model: :class:`Model`
    :param int num_patches: Number of patches to extract.
    :param bool normalize: Whether to normalize each C1 patch.

    """
    self.model, self.num_patches, self.normalize = model, num_patches, normalize

  def __call__(self, state):
    """Transform an input model state to a set of C1 patches.

    :param state: Input model state.
    :type state: :class:`State`
    :returns: list of (patch, sample location) pairs

    """
    return self.model.SampleC1Patches(self.num_patches, state, self.normalize)
