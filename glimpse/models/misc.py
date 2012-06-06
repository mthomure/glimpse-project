"""General functions that are applicable to multiple models."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import copy
import Image
from itertools import chain, count, islice
import logging
from math import sqrt
import numpy as np
import random

from glimpse import backends
from glimpse.backends import InsufficientSizeException
from glimpse import pools
from glimpse.util import ImageToArray, ACTIVATION_DTYPE, TypeName

class LayerSpec(object):
  """Describes a single layer in a model."""

  #: A unique (within a given model) identifier for the layer. Not necessarily
  #: numeric.
  ident = 0

  #: A user-friendly name for the layer.
  name = ""

  #: Layers whose data is required to compute this layer.
  depends = []

  def __init__(self, ident, name = None, *depends):
    self.ident, self.name, self.depends = ident, name, depends

  def __str__(self):
    return self.name

  def __repr__(self):
    return "LayerSpec(ident=%s, name=%s, depends=[%s])" % (self.ident,
        self.name, ", ".join(map(str, self.depends)))

  def __eq__(self, other):
    # Note that this is only useful for comparing layers for a single model
    # (since it only uses the `ident` attribute).
    return isinstance(other, LayerSpec) and self.ident == other.ident

class InputSourceLoadException(Exception):
  """Thrown when an input source can not be loaded."""

  def __init__(self, msg = None, source = None):
    super(InputSourceLoadException, self).__init__(msg)
    #: The *source* of the exception. This is generally an :class:`InputSource`
    #: object.
    self.source = source

  def __str__(self):
    return "InputSourceLoadException(%s)" % self.source

  __repr__ = __str__

class InputSource(object):
  """Describes the input to a hierarchical model.

  Example inputs could include the path to a single image, or the path and frame
  of a video.

  Example:

  Load a still image from disk, and resize its shorter edge to 128 pixels
  (maintaining aspect ratio).

  >>> source = InputSource(image_path = "/tmp/MyImage.jpg", resize = 128)

  """

  #: Path to an image file.
  image_path = None

  #: If set, the image is resized such that its shorter edge has this length.
  #: The resize operation preserves the aspect ratio of the image.
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
      raise InputSourceLoadException("I/O error while loading image",
          source = self)
    if self.resize != None:
      width, height = img.size
      ratio = float(self.resize) / min(width, height)
      new_size = int(width * ratio), int(height * ratio)
      logging.info("Resize image %s from (%d, %d) to %s", self.image_path,
          width, height, new_size)
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

class DependencyError(Exception):
  """Indicates that a dependency required to build a node in the network is
  unavailable."""

class BaseLayer(object):
  """Enumerator for model layers."""

  #: Specifier for the source of the image data. This may be :obj:`None` if the
  #: data was generated programatically, or an :class:`InputSource` object if
  #: the data came from an image file.
  SOURCE = LayerSpec("s", "source")

  #: Specifier for the raw input data.
  IMAGE = LayerSpec("i", "image", SOURCE)

  @classmethod
  def FromId(cls, ident):
    """Lookup a LayerSpec object by ID.

    :param ident: Model-unique layer identifier.
    :rtype: :class:`LayerSpec`

    Example:

    >>> lyr = BaseLayer.FromId(BaseLayer.IMAGE.ident)
    >>> assert(lyr == BaseLayer.IMAGE)

    """
    if isinstance(ident, LayerSpec):
      return ident
    for layer in cls.AllLayers():
      if layer.ident == ident:
        return layer
    raise ValueError("Unknown layer id: %r" % ident)

  @classmethod
  def FromName(cls, name):
    """Lookup a LayerSpec object by name.

    This method is not case sensitive.

    :param str name: Layer name.
    :rtype: :class:`LayerSpec`

    Example:

    >>> lyr = BaseLayer.FromName(BaseLayer.IMAGE.name)
    >>> assert(lyr == BaseLayer.IMAGE)

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

    >>> assert(BaseLayer.IMAGE in BaseLayer.AllLayers())

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

    >>> assert(BaseLayer.IsSublayer(BaseLayer.SOURCE, BaseLayer.IMAGE))
    >>> assert(not BaseLayer.IsSublayer(BaseLayer.IMAGE, BaseLayer.SOURCE))

    """
    for lyr in super_layer.depends:
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

    >>> assert(BaseLayer.TopLayer() == BaseLayer.IMAGE)

    """
    all_layers = cls.AllLayers()
    for layer in all_layers:
      if any(layer in l.depends for l in all_layers):
        continue
      return layer
    raise Exception("Internal error: Failed to find top layer in network")

class BaseState(dict):
  """A dictionary container for the :class:`Model` state.

  The main purpose of extending the dictionary (instead of just storing states
  as dictionary objects) is to indicate with which model a given state is
  associated. Each model has a seperate state object, so it is always clear
  which model generated a given state object.

  """

  #: The datatype associated with the model for this state. Should be set by
  #: sub-class.
  ModelClass = None

  def __getitem__(self, name):
    """Lookup activation for a given layer.

    :param name: Layer identifier.
    :type name: scalar or :class:`LayerSpec`

    """
    if isinstance(name, LayerSpec):
      name = name.ident
    return super(BaseState, self).__getitem__(name)

class BaseModel(object):
  """Abstract base class for a Glimpse model."""

  #: The datatype associated with layer descriptors for this model. This should
  #: be over-ridden by the sub-class with a descendent of :class:`BaseLayer`.
  LayerClass = BaseLayer

  #: The type of the parameter collection associated with this model. This
  #: should be over-ridden by the sub-class.
  ParamClass = object

  #: The datatype associated with network states for this model. This should be
  #: over-ridden by the sub-class, and should generally be a descendent of
  #: :class:`BaseState`.
  StateClass = BaseState

  def __init__(self, backend = None, params = None):
    """Create new object.

    :param backend: Implementation of backend operations, such as dot-products.
    :param params: Model configuration.

    """
    if backend == None:
      backend = backends.MakeBackend()
    else:
      backend = copy.copy(backend)
    if params == None:
      params = self.ParamClass()
    else:
      if not isinstance(params, self.ParamClass):
        raise ValueError("Params object has wrong type: expected %s, got %s" % \
            (self.ParamClass, type(params)))
      params = copy.copy(params)
    self.backend = backend
    self.params = params

  def _BuildSingleNode(self, output_id, state):
    """Internal: Compute activity for a single model layer.

    This method is called by :meth:`_BuildNode`.

    :param output_id: Layer to compute.
    :type output_id: :class:`LayerSpec`
    :param state: Model state from which to compute given layer.
    :type state: StateClass
    :returns: Activity for given output layer.
    :rtype: ndarray

    """
    lyr = self.LayerClass
    if output_id == lyr.SOURCE.ident:
      raise DependencyError
    elif output_id == lyr.IMAGE.ident:
      return self.BuildImageFromInput(state[lyr.SOURCE.ident].CreateImage())
    raise ValueError("Unknown layer ID: %r" % output_id)

  def _BuildNode(self, output_id, state):
    """Internal: Construct the given output value.

    The output value is built by recursively building required dependencies. A
    node is considered to be *built* if the node's key is set in the state
    dictionary, even if the corresponding value is None.

    This method is called by :meth:`BuildLayer`.

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
      layer = self.LayerClass.FromId(output_id)
      for node in layer.depends:
        state = self._BuildNode(node.ident, state)
      # Compute the output node
      state[output_id] = self._BuildSingleNode(output_id, state)
    return state

  def BuildLayer(self, output_layer, state, save_all = True):
    """Apply the model through to the given layer.

    :param output_layer: Output layer to compute. If scalar is given, this
       should be the ID of the desired layer.
    :type output_layer: :class:`LayerSpec` or scalar
    :param state: Initial model state from which to compute the output layer.
    :type state: StateClass
    :param bool save_all: Whether the resulting state should contain values for
       all computed layers in the network, or just the output layer. Note that
       source information is always preserved.
    :returns: Output state containing the given layer
    :rtype: StateClass

    Examples:

    Get the :attr:`IMAGE <BaseLayer.IMAGE>` layer for an image.

    >>> model = BaseModel()
    >>> input_state = model.MakeStateFromFilename(glab.GetExampleImage())
    >>> output_state = model.BuildLayer(BaseLayer.IMAGE, input_state)
    >>> assert(BaseLayer.IMAGE.ident in output_state)

    """
    state = copy.copy(state)  # get a shallow copy of the model state
    if isinstance(output_layer, LayerSpec):
      output_layer = output_layer.ident
    if output_layer in state:
      return state
    try:
      state = self._BuildNode(output_layer, state)
    except InsufficientSizeException, ex:
      # Try to annotate exception with source information.
      ex.source = state.get(self.LayerClass.SOURCE.ident, None)
      raise ex
    if not save_all:
      state_ = self.StateClass()
      # Keep output layer data
      state_[output_layer] = state[output_layer]
      # Keep source information
      src = self.LayerClass.SOURCE.ident
      if src in state:
        state_[src] = state[src]
      state = state_
    return state

  def BuildLayerCallback(self, output_layer, save_all = True):
    """Create a function that will apply the model through to the given layer.

    :param output_layer: Output layer to compute. If scalar is given
    :type output_layer: :class:`LayerSpec` or scalar
    :param state: Initial model state from which to compute the output layer.
    :type state: StateClass
    :param bool save_all: Whether the resulting state should contain values for
       all computed layers in the network, or just the output layer.
    :returns: A callable object that, when evaluated, will apply the model.
    :rtype: :class:`LayerBuilder`

    See also :meth:`BuildLayer`.

    Examples:

    Get the IMAGE layer for a set of images.

    >>> model = BaseModel()
    >>> builder = model.BuildLayerCallback(BaseLayer.IMAGE)
    >>> images = glab.GetExampleImages()
    >>> input_states = map(model.MakeStateFromFilename, images)
    >>> output_states = map(builder, input_states)
    >>> assert(len(output_states) == len(images))
    >>> assert(all(BaseLayer.IMAGE.ident in state for state in output_states))

    """
    if isinstance(output_layer, LayerSpec):
      output_layer = output_layer.ident
    return LayerBuilder(self, output_layer, save_all)

  def MakeStateFromFilename(self, filename, resize = None):
    """Create a model state with a single SOURCE layer.

    :param str filename: Path to an image file.
    :param int resize: Scale minimum edge to fixed length.
    :returns: The new model state.
    :rtype: :class:`BaseState`

    Examples:

    Create a model state from an image path.

    >>> model = BaseModel()
    >>> image = glab.GetExampleImage()
    >>> state = model.MakeStateFromImage(image)

    """
    state = self.StateClass()
    state[self.LayerClass.SOURCE.ident] = InputSource(filename, resize = resize)
    return state

  def MakeStateFromImage(self, image):
    """Create a model state with a single IMAGE layer.

    :param image: Input data. If array, values should lie in the range [0, 1].
    :type image: PIL.Image or 2D ndarray
    :param int resize: Scale minimum edge to fixed length.
    :returns: The new model state.
    :rtype: StateClass

    Examples:

    Create a model state from a PIL image object.

    >>> model = BaseModel()
    >>> image = Image.open(glab.GetExampleImage())
    >>> state = model.MakeStateFromImage(image)

    Create a model state from an array of pixel data.

    >>> model = BaseModel()
    >>> image = numpy.random.randint(0, 256, (100, 100))
    >>> state = model.MakeStateFromImage(image)

    """
    state = self.StateClass()
    state[self.LayerClass.IMAGE.ident] = self.BuildImageFromInput(image)
    return state

  def SamplePatches(self, layer, num_patches, state, normalize = False):
    """Sample patches from the given layer for a single image.

    Patches are sampled from random locations and scales.

    :param layer: Layer from which to extract patches. If scalar is given, this
       should be the ID of the desired layer.
    :type layer: :class:`LayerSpec` or scalar
    :param num_patches: Number of patches to extract at each size, in the format
       ``( (patch_size1, count1), ..., (patch_sizeN, countN) )``.
    :type num_patches: tuple of pairs of int
    :param state: Input state for which layer activity is computed.
    :type state: StateClass
    :param bool normalize: Whether to normalize each patch.
    :returns: 2D list of (patch, location) pairs. The list axes correspond to
       kernel size and kernel offset, respectively. The location is a triple,
       whose axes correspond to the scale, y-offset, and x-offset of the patch.
    :rtype: 2D list of (patch, location) pairs

    Examples:

    Get 10 patches from an image.

    >>> model = BaseModel()
    >>> num_patches = 10
    >>> image = glab.GetExampleImage()
    >>> state = model.MakeStateFromFilename(image)
    >>> results = model.SamplePatches(BaseLayer.IMAGE, num_patches, state)
    >>> assert(len(results) == num_patches)

    """
    if isinstance(layer, LayerSpec):
      layer = layer.ident
    state = self.BuildLayer(layer, state)
    data = state[layer]

    def GetPatches(patch_width, num_patches):
      try:
        patch_it = PatchGenerator(data, patch_width)
        patches = list(islice(patch_it, num_patches))
      except InsufficientSizeException, ex:
        # Try to annotate exception with source information.
        ex.source = state.get(self.LayerClass.SOURCE.ident, None)
        raise ex
      # TEST CASE: single state with uniform C1 activity and using
      # normalize=True, check that result does not contain NaNs.
      if normalize:
        for patch, _ in patches:
          norm = np.linalg.norm(patch)
          if norm == 0:
            logging.warn("Normalizing empty patch")
            patch[:] = 1.0 / sqrt(patch.size)
          else:
            patch /= norm
      return patches

    return [ GetPatches(w, c) for w, c in num_patches ]

  def SamplePatchesCallback(self, layer, num_patches, normalize = False):
    """Create a function that will sample patches.

    :param layer: Layer from which to extract patches. If scalar is given, this
       should be the ID of the desired layer.
    :param num_patches: Number of patches to extract at each size, in the format
       ``( (patch_size1, count1), ..., (patch_sizeN, countN) )``.
    :type num_patches: tuple of pairs of int
    :param bool normalize: Whether to normalize each patch.
    :return: A callable object that, when evaluated, will sample patches
    :rtype: :class:`PatchSampler`

    Examples:

    Get 10 patches from each image in a list.

    >>> model = BaseModel()
    >>> num_patches_per_image = 10
    >>> sampler = model.SamplePatchesCallback(model, BaseLayer.IMAGE,
            num_patches_per_image)
    >>> images = glab.GetExampleImages()
    >>> states = map(model.MakeStateFromFilename, images)
    >>> results_per_image = map(sampler, states)
    >>> assert(len(results_per_image) == len(images))
    >>> assert(all(len(results) == num_patches_per_image
            for results in results_per_image))

    """
    return PatchSampler(self, layer, num_patches, normalize)

  def BuildImageFromInput(self, input_):
    """Create the initial image layer from some input.

    :param input_: Input data. If array, values should lie in the range [0, 1].
    :type input_: PIL.Image or 2D ndarray
    :returns: image layer data.
    :rtype: 2D ndarray of float

    """
    return ImageLayerFromInputArray(input_, self.backend)

class LayerBuilder(object):
  """Represents a serializable function.

  This function computes a network state transformation, computing the value of
  some network layer and any required dependencies.

  """

  #: Set the name of the class. This is needed for use with IPython.parallel.
  __name__ = "LayerBuilder"

  def __init__(self, model, output_layer_id, save_all = True):
    """Create a new object.

    :param model: The model to use when computing layer activity.
    :type model: :class:`BaseModel`
    :param output_layer_id: Highest layer in model to compute.
    :type output_layer_id: :class:`LayerSpec` or scalar (layer id)
    :param bool save_all: Whether the resulting state should contain values for
       all computed layers in the network, or just the output layer.

    """
    self.model, self.output_layer_id, self.save_all = model, output_layer_id, \
        save_all

  def __call__(self, state):
    """Transform between network states.

    :param state: Initial model state from which to compute the output layer.
    :type state: StateClass
    :returns: Output state containing the given layer
    :rtype: model.StateClass

    See also :meth:`BaseModel.BuildLayer`.

    """
    return self.model.BuildLayer(self.output_layer_id, state, self.save_all)

  def __str__(self):
    return "%s(model=%s, output_layer_id=%s, save_all=%s)" % \
        (TypeName(self), type(self.model), self.output_layer_id, self.save_all)

  __repr__ = __str__

class PatchSampler(object):
  """Represents a serializable function that can be called repeatedly for
  different input states.

  This function computes activity for the desired layer and then extracts
  patches from randomly-sampled locations and scales.

  """

  #: Set the name of the class. This is needed for use with IPython.parallel.
  __name__ = "PatchSampler"

  def __init__(self, model, layer, num_patches, normalize = False):
    """Create new object.

    :param model: Model instantiation to use when computing activity.
    :type: model: :class:`BaseModel`
    :param layer: Layer from which to extract patches. If scalar is given, this
       should be the ID of the desired layer.
    :type layer: :class:`LayerSpec` or scalar
    :param num_patches: Number of patches to extract at each size, in the format
       ``( (patch_size1, count1), ..., (patch_sizeN, countN) )``.
    :type num_patches: tuple of pairs of int
    :param bool normalize: Whether to normalize each C1 patch.

    """
    self.model, self.layer, self.num_patches, self.normalize = model, layer, \
        num_patches, normalize

  def __call__(self, state):
    """Transform an input model state to a set of output layer patches.

    :param state: Input model state.
    :type state: model.StateClass
    :returns: 2D list of (patch, location) pairs. The list axes correspond to
       kernel size and kernel offset, respectively. The location is a triple,
       whose axes correspond to the scale, y-offset, and x-offset of the patch.
    :rtype: 2D list of pairs

    See also :meth:`BaseModel.SamplePatches`.

    """
    return self.model.SamplePatches(self.layer, self.num_patches, state,
        self.normalize)

  def __str__(self):
    return "%s(model=%s, layer=%s, num_patches=%s, normalize=%s)" % \
        (TypeName(self), type(self.model), self.layer, self.num_patches,
        self.normalize)

  __repr__ = __str__

def PatchGenerator(data, patch_width):
  """Sample patches from a layer of activity.

  :param data: Activity maps, one map per scale.
  :type data: ND ndarray, or list of (N-1)D ndarray. Must have N >= 3.
  :param int patch_width: Spatial extent of patch.
  :returns: Infinite iterator over patch arrays and corresponding locations
     (i.e., iterator elements are 2-tuples). Location gives top-left corner of
     region.

  Examples:

  Extract 2D patches from a 3D array:

  >>> shape = (4, 100, 100)
  >>> data = numpy.random.random(shape)
  >>> patch_iterator = PatchGenerator(data, patch_width = 10)

  Note that `patch_iterator` is an (infinite) iterator, not a list. The
  following will get the first 50 patches as a list.

  >>> patches = list(itertools.islice(patch_iterator, 50))

  """
  assert len(data) > 0
  assert all(d.ndim > 1 for d in data)
  num_scales = len(data)
  while True:
    scale = random.randint(0, num_scales - 1)
    data_scale = data[scale]
    num_bands = data_scale.ndim - 2
    layer_height, layer_width = data_scale.shape[-2:]
    if layer_height <= patch_width or layer_width <= patch_width:
      raise InsufficientSizeException("Layer must be larger than patch size.")
    # Choose the top-left corner of the region.
    y0 = random.randint(0, layer_height - patch_width)
    x0 = random.randint(0, layer_width - patch_width)
    # Copy data from all bands in the given X-Y region.
    index = [ slice(None) ] * num_bands
    index += [ slice(y0, y0 + patch_width), slice(x0, x0 + patch_width) ]
    patch = data_scale[ index ]
    yield patch.copy(), (scale, y0, x0)

def ImprintKernels(model, sample_layer, kernel_sizes, num_kernels,
    input_states, normalize = True, pool = None):
  """Create a set of kernels by sampling patches from the layer below.

  :param model: Glimpse model with the property `s2_kernel_sizes`.
  :type model: :class:`BaseModel`
  :param sample_layer: Layer from which to sample.
  :type sample_layer: str or :class:`LayerSpec`
  :param kernel_sizes: Kernel widths to support.
  :type kernel_sizes: list of int
  :param int num_kernels: Number of patches to sample for each width.
  :param input_states: Initial network states from which to compute layer
     activity.
  :type input_states: list of `model.StateClass`
  :param bool normalize: Whether to scale each kernel to have unit norm.
  :param pool: Worker pool used to evaluate the model.
  :returns: Kernels, and their corresponding locations. Kernels are returned as
     a list of (N+1)-dimensional arrays, where N is the number of axes in a
     single kernel. The list axis and first axis of the array correspond to the
     kernel size and kernel offset, respectively.  Locations are returned as a
     2D list of 4-tuples, where list axes correspond to kernel size and kernel
     offset, respectively. Each location is given as a 4-tuple, with elements
     corresponding to the input state index, scale, y-offset, and x-offset of
     the corresponding kernel.

  Examples:

  >>> model = BaseModel()
  >>> images = glab.GetExampleImages()
  >>> states = map(model.MakeStateFromFilename, images)
  >>> kernels, locations = ImprintKernels(model, kernel_sizes = (7, 11),
          num_kernels = 10, states)

  The result should contain a sub-list for each of the two kernel sizes.

  >>> assert len(kernels) == 2
  >>> assert len(locations) == 2
  >>> assert all(len(ks) == 10 for ps in kernels)
  >>> assert all(len(ls) == 10 for ls in locations)

  """
  if pool == None:
    pool = pools.MakePool()
  if num_kernels < len(input_states):
    # Take a random subset of images.
    random.shuffle(input_states)
    input_states = input_states[:num_kernels]
    num_imprinted_kernels = 1
  else:
    num_imprinted_kernels, extra = divmod(num_kernels, len(input_states))
    if extra > 0:
      num_imprinted_kernels += 1
  # The SamplePatchesCallback() function takes a mapping of patch size
  # information, where the key is a patch size and the value gives the number of
  # patches for that size (for each image). We create that mapping here.
  kernels_per_image = [ (size, num_imprinted_kernels) for size in kernel_sizes ]
  # Create a callback, which we can pass to the worker pool.
  sampler = model.SamplePatchesCallback(sample_layer, kernels_per_image,
      normalize = normalize)
  # Compute layer activity, and sample patches.
  values_per_image = pool.map(sampler, input_states)
  # We now have an iterator over (kernel, location) pairs for each image.
  # Evaluate the iterator, and store the result as a list. Note that we must
  # evaluate before concatenating (below) to ensure we don't leave requests
  # sitting on the worker pool.
  values_per_image = map(list, values_per_image)
  # Sanity check the number of returned results.
  assert len(input_states) == len(values_per_image), \
      "Expected %d sampled patch sets, but got %d" % (len(input_states),
          len(values_per_image))
  # Add input state index to locations. Note that the sampler returns a 2D list
  # of pairs for each image. The list axes correspond to kernel size and kernel
  # offset, respectively. The pair contains a patch array and location,
  # respectively. The location is a triple, whose axes correspond to the scale,
  # y-offset, and x-offset of the patch, respectively.
  values_per_image = [ [ [
      (kernel, (image_idx,) + loc) for kernel, loc in values ]
      for values in values_per_ksize ]
      for values_per_ksize, image_idx in zip(values_per_image, count()) ]
  # Gather kernels across images, and arrange by size.
  values_per_ksize = zip(*values_per_image)
  # Each kernel size bin contains a 2D list of entries, where axes correspond to
  # input image and kernel offset. Flatten this to a 1D list.
  values_per_ksize = [ list(chain(*list2d)) for list2d in values_per_ksize ]
  all_kernels, all_locations = [], []  # Result lists not grouped by image.
  # Loop over kernel sizes.
  for values in values_per_ksize:
    # We may have sampled too many kernels (e.g., if the number of requested
    # patches is not an even multiple of the number of images). If so, crop
    # the results.
    values = values[:num_kernels]
    # Convert list of tuples to tuple of lists.
    kernels, locations = zip(*values)
    # Convert set of kernels to a single numpy array.
    kernels = np.array(kernels, ACTIVATION_DTYPE)
    all_kernels.append(kernels)
    all_locations.append(locations)
  return all_kernels, all_locations

def Whiten(data):
  """Normalize an array, such that each location contains equal energy.

  For each X-Y location, the vector :math:`a` of data (containing activation for
  each band) is *sphered* according to:

  .. math::

    a' = (a - \mu_a ) / \sigma_a

  where :math:`\mu_a` and :math:`\sigma_a` are the mean and standard deviation
  of :math:`a`, respectively.

  .. caution::

     This function modifies the input data in-place.

  :param data: Layer activity to modify.
  :type data: 3D ndarray of float
  :returns: The `data` array.
  :rtype: 3D ndarray of float

  """
  assert data.ndim == 3
  data -= data.mean(0)
  norms = np.sqrt((data**2).sum(0))
  norms[ norms < 1 ] = 1
  data /= norms
  return data

def GetModelClass(name = None):
  """Lookup a Glimpse model class by name.

  :param str name: The name of the model. This corresponds to the model's
     package name. The default is read from the :envvar:`GLIMPSE_MODEL`
     environment variable, or is ``hmax`` if this is not set.

  Examples:

  Create a new instance of the 'viz2' model:

  >>> ModelClass = GetModelClass("viz2")
  >>> assert(ModelClass == glimpse.viz2.Model)

  """
  import os
  if name == None:
    name = os.environ.get('GLIMPSE_MODEL', 'viz2')
  pkg = __import__("glimpse.models.%s" % name, globals(), locals(), ['Model'],
      0)
  try:
    return getattr(pkg, 'Model')
  except AttributeError:
    raise ValueError("Unknown model name: %s" % name)
