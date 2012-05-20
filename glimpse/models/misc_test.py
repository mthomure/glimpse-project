
import os
import unittest

from glimpse import glab
from glimpse.models.misc import LayerSpec, InputSource, \
    InputSourceLoadException, BaseLayer, BaseState, BaseModel, \
    DependencyError, ImprintS2Prototypes

EXAMPLE_IMAGE = os.path.join(glab.GetExampleCorpus(), 'cats', 'Marcus_bed.jpg')
EXAMPLE_IMAGE2 = os.path.join(glab.GetExampleCorpus(), 'dogs',
    '41-27Monate1.JPG')
EXAMPLE_IMAGE_BAD = '/invalid-file-name.jpg'

class Layer(BaseLayer):

  RESHAPED_IMAGE = LayerSpec("r", "reshape-image", BaseLayer.IMAGE)

class Model(BaseModel):

  LayerClass = Layer

  ParamClass = object

  StateClass = BaseState

  def _BuildSingleNode(self, output_id, state):
    if output_id == Layer.RESHAPED_IMAGE.id:
      data = state[Layer.IMAGE.id]
      return data.reshape((1,) + data.shape)
    return super(Model, self)._BuildSingleNode(output_id, state)

class S2Layer(BaseLayer):

  C1 = LayerSpec("c1", "C1", BaseLayer.IMAGE)

class S2Model(BaseModel):

  LayerClass = S2Layer

  ParamClass = object

  StateClass = BaseState

  @property
  def s2_kernel_sizes(self):
    return (2, 3)

  def _BuildSingleNode(self, output_id, state):
    if output_id == S2Layer.C1.id:
      data = state[S2Layer.IMAGE.id]
      # Return the array with one scale and one band.
      return data.reshape((1, 1) + data.shape)
    return super(S2Model, self)._BuildSingleNode(output_id, state)

class TestBaseLayer(unittest.TestCase):

  def testFromId(self):
    lyr = Layer.RESHAPED_IMAGE
    self.assertEqual(Layer.FromId(lyr.id), lyr)

  def testFromName(self):
    lyr = Layer.RESHAPED_IMAGE
    self.assertEqual(Layer.FromName(lyr.name), lyr)

  def testAllLayers(self):
    expected_layers = (Layer.SOURCE, Layer.IMAGE, Layer.RESHAPED_IMAGE)
    layers = Layer.AllLayers()
    self.assertEqual(len(layers), 3)
    self.assertEqual(sorted(layers), sorted(expected_layers))

  def testIsSublayer(self):
    self.assertTrue(Layer.IsSublayer(Layer.IMAGE, Layer.RESHAPED_IMAGE))
    self.assertFalse(Layer.IsSublayer(Layer.RESHAPED_IMAGE, Layer.SOURCE))

  def testTopLayer(self):
    self.assertEqual(Layer.TopLayer(), Layer.RESHAPED_IMAGE)
    self.assertEqual(BaseLayer.TopLayer(), BaseLayer.IMAGE)

class TestInputSource(unittest.TestCase):

  def testCreateImage(self):
    source = InputSource(EXAMPLE_IMAGE)
    img = source.CreateImage()
    self.assertNotEqual(img, None)
    self.assertEqual(img.size, (256, 256))
    self.assertEqual(img.fp.name, EXAMPLE_IMAGE)

  def testCreateImage_throwsLoadException(self):
    source = InputSource(EXAMPLE_IMAGE_BAD)
    self.assertRaises(InputSourceLoadException, source.CreateImage)

  def testEqual(self):
    source1 = InputSource(EXAMPLE_IMAGE)
    source2 = InputSource(EXAMPLE_IMAGE)
    self.assertEqual(source1, source2)

class TestBaseModel(unittest.TestCase):

  def testBuildSingleNode(self):
    # Model can't build source layer. This should raise an exception.
    model = Model()
    state = BaseState()
    self.assertRaises(DependencyError, model._BuildSingleNode, Layer.SOURCE.id,
        state)

  def testBuildNode(self):
    # Model should raise exception on missing dependencies.
    model = Model()
    state = BaseState()
    self.assertRaises(DependencyError, model._BuildNode, Layer.SOURCE.id, state)
    self.assertRaises(DependencyError, model._BuildNode, Layer.IMAGE.id, state)

  def _testBuildLayer(self, use_callback):
    model = Model()
    state = model.MakeStateFromFilename(EXAMPLE_IMAGE)
    layer = Layer.RESHAPED_IMAGE.id
    if use_callback:
      callback = model.BuildLayerCallback(layer)
      out_state = callback(state)
    else:
      out_state = model.BuildLayer(layer, state)
    self.assertNotEqual(out_state, None)
    self.assertTrue(layer in out_state)
    result = out_state[layer]
    self.assertNotEqual(result, None)
    self.assertEqual(result.shape, (1, 256, 256))

  def testBuildLayer_failsOnEmptyState(self):
    model = Model()
    state = BaseState()
    def f():
      return model.BuildLayer(Layer.IMAGE, state)
    self.assertRaises(DependencyError, f)

  def testBuildLayer(self):
    self._testBuildLayer(use_callback = False)

  def testBuildLayerCallback(self):
    self._testBuildLayer(use_callback = True)

  def testBuildLayer_failsOnBadImage(self):
    model = Model()
    state = model.MakeStateFromFilename(EXAMPLE_IMAGE_BAD)
    layer = Layer.IMAGE.id
    self.assertRaises(InputSourceLoadException, model.BuildLayer, layer, state)

  def testBuildImageLayerFromPath(self):
    lyr = Layer.IMAGE
    model = Model()
    state = model.MakeStateFromFilename(EXAMPLE_IMAGE)
    out_state = model.BuildLayer(model, lyr, state)
    result = out_state[lyr.id]
    img = Image.open(EXAMPLE_IMAGE)
    self.assertEqual(result.shape, img.size[::-1])

  def _testSamplePatches(self, use_callback):
    model = Model()
    state = model.MakeStateFromFilename(EXAMPLE_IMAGE)
    # Sample 10 patches from the image layer at sizes of 6x6 and 10x10.
    num_patches_per_size = 10
    sizes = (4, 6)
    num_patches = [ (size, num_patches_per_size) for size in sizes ]
    if use_callback:
      callback = model.SamplePatchesCallback(Layer.RESHAPED_IMAGE,
          num_patches)
      values = callback(state)
    else:
      values = model.SamplePatches(Layer.RESHAPED_IMAGE, num_patches, state)
    self.assertEqual(len(values), len(sizes))
    # Each prototype should have correct size.
    for idx in range(len(sizes)):
      values_for_size = values[idx]
      self.assertEqual(len(values_for_size), num_patches_per_size)
      self.assertTrue(all(proto.shape == (sizes[idx], sizes[idx])
          for proto, loc in values_for_size))

  def testSamplePatches(self):
    self._testSamplePatches(use_callback = False)

  def testSamplePatchesCallback(self):
    self._testSamplePatches(use_callback = True)

class TestImprint(unittest.TestCase):

  def testImprintS2Prototypes(self):
    model = S2Model()
    states = map(model.MakeStateFromFilename, (EXAMPLE_IMAGE, EXAMPLE_IMAGE2))
    num_protos = 10
    prototypes, locations = ImprintS2Prototypes(model, num_protos, states)
    num_sizes = len(model.s2_kernel_sizes)
    self.assertEqual(len(prototypes), num_sizes)
    for protos_for_size, size in zip(prototypes, model.s2_kernel_sizes):
      self.assertEqual(protos_for_size.shape, (num_protos, 1, size, size))
    self.assertEqual(len(locations), num_sizes)
    for locs_for_size in locations:
      self.assertTrue(all(len(loc) == 4 for loc in locs_for_size))
