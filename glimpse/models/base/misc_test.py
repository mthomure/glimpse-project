import Image
import itertools
import numpy as np
import unittest

from glimpse.util.garray import toimage, fromimage
from glimpse.backends import InputSizeError, BackendError
from .model import Model, Layer, State, layer_builder
from .layer import LayerSpec
from .param import *
from .misc import *

class PrepareImageTest(unittest.TestCase):

  def test_noResize(self):
    data = np.random.random((100,101)) * 255
    data = data.astype(np.uint8)
    img = toimage(data)
    params = Params(image_resize_method = ResizeMethod.NONE)
    img2 = PrepareImage(img, params)
    self.assertEqual(img, img2)

  def test_scaleShortEdge(self):
    img = Image.new('L', (100,200))
    params = Params(image_resize_method = ResizeMethod.SCALE_SHORT_EDGE,
        image_resize_length = 256)
    img2 = PrepareImage(img, params)
    self.assertSequenceEqual(img2.size, (256, 512))

  def test_scaleLongEdge(self):
    img = Image.new('L', (100,200))
    params = Params(image_resize_method = ResizeMethod.SCALE_LONG_EDGE,
        image_resize_length = 256)
    img2 = PrepareImage(img, params)
    self.assertSequenceEqual(img2.size, (128, 256))

  def test_scaleWidth(self):
    img = Image.new('L', (100,200))
    params = Params(image_resize_method = ResizeMethod.SCALE_WIDTH,
        image_resize_length = 256)
    img2 = PrepareImage(img, params)
    self.assertSequenceEqual(img2.size, (256, 512))

  def test_scaleHeight(self):
    img = Image.new('L', (100,200))
    params = Params(image_resize_method = ResizeMethod.SCALE_HEIGHT,
        image_resize_length = 256)
    img2 = PrepareImage(img, params)
    self.assertSequenceEqual(img2.size, (128, 256))

  def test_scaleAndCrop_scaleAndNoCrop(self):
    img = Image.new('L', (100,200))
    params = Params(image_resize_method = ResizeMethod.SCALE_AND_CROP,
        image_resize_length = 256, image_resize_aspect_ratio = .5)
    img2 = PrepareImage(img, params)
    self.assertSequenceEqual(img2.size, (256, 512))

  def test_scaleAndCrop_cropAndNoScale(self):
    img = Image.new('L', (100,200))
    params = Params(image_resize_method = ResizeMethod.SCALE_AND_CROP,
        image_resize_length = 100, image_resize_aspect_ratio = 1)
    img2 = PrepareImage(img, params)
    self.assertSequenceEqual(img2.size, (100, 100))

  def test_scaleAndCrop_doBoth(self):
    img = Image.new('L', (100,200))
    params = Params(image_resize_method = ResizeMethod.SCALE_AND_CROP,
        image_resize_length = 256, image_resize_aspect_ratio = .25)
    img2 = PrepareImage(img, params)
    self.assertSequenceEqual(img2.size, (256, 1024))

class ReshapedLayer(Layer):

  RESHAPED_IMAGE = LayerSpec("r", "reshape-image", [Layer.IMAGE])
  C1 = LayerSpec("c1", "C1", [Layer.IMAGE])

INVALID_LAYER_SPEC = LayerSpec("invalid", "invalid-layer")

class ReshapedModel(Model):

  LayerClass = ReshapedLayer

  @layer_builder(ReshapedLayer.RESHAPED_IMAGE)
  def ReshapeImage(self, img):
    return img.reshape((1,) + img.shape)

  @layer_builder(ReshapedLayer.C1)
  def BuildC1(self, data):
    # Return the array with one scale and one band.
    return data.reshape((1, 1) + data.shape)

class BrokenModel(ReshapedModel):

  @layer_builder(ReshapedLayer.RESHAPED_IMAGE)
  def ReshapeImage(self, img):
    raise BackendError

class BuildLayerTest(unittest.TestCase):

  def test_saveAll(self):
    m = ReshapedModel()
    img = Image.new('L', (100,100))
    s = BuildLayer(m, ReshapedLayer.RESHAPED_IMAGE, m.MakeState(img),
        save_all = True)
    self.assertIn(ReshapedLayer.IMAGE, s)
    self.assertIn(ReshapedLayer.RESHAPED_IMAGE, s)

  def test_saveAllIsFalse(self):
    m = ReshapedModel()
    img = Image.new('L', (100,100))
    s = BuildLayer(m, ReshapedLayer.RESHAPED_IMAGE, m.MakeState(img),
        save_all = False)
    self.assertNotIn(ReshapedLayer.IMAGE, s)
    self.assertIn(ReshapedLayer.RESHAPED_IMAGE, s)

  def test_backendError(self):
    m = BrokenModel()
    img = Image.new('L', (100,100))
    with self.assertRaises(BackendError):
      BuildLayer(m, ReshapedLayer.RESHAPED_IMAGE, m.MakeState(img))

  def test_missingSource(self):
    m = ReshapedModel()
    with self.assertRaises(ValueError):
      BuildLayer(m, ReshapedLayer.RESHAPED_IMAGE, State())

  def test_badLayer(self):
    m = Model()
    with self.assertRaises(ValueError):
      BuildLayer(m, INVALID_LAYER_SPEC, State())

  def test_multipleLayers(self):
    m = ReshapedModel()
    img = Image.new('L', (100,100))
    s = BuildLayer(m, (ReshapedLayer.RESHAPED_IMAGE, ReshapedLayer.C1),
        m.MakeState(img))
    self.assertIn(ReshapedLayer.RESHAPED_IMAGE, s)
    self.assertIn(ReshapedLayer.C1, s)

class SamplePatchesTest(unittest.TestCase):

  def test_noSamples(self):
    model = ReshapedModel()
    img = Image.new('L', (100,200))
    num_patches = 0
    patch_size = 6
    patches, locs = SamplePatches(model, ReshapedLayer.C1, num_patches,
        patch_size, model.MakeState(img))
    self.assertEqual(len(patches), num_patches)
    self.assertEqual(len(locs), num_patches)

  def test_endToEnd(self):
    model = ReshapedModel(Params(image_resize_method = ResizeMethod.NONE))
    width = 100
    height = 200
    img = Image.new('L', (width, height))
    num_patches = 10
    patch_size = 6
    patches, locs = SamplePatches(model, ReshapedLayer.C1, num_patches,
        patch_size, model.MakeState(img))
    self.assertEqual(len(patches), num_patches)
    self.assertEqual(len(locs), num_patches)
    for patch, loc in zip(patches, locs):
      self.assertEqual(patch.ndim, 3)
      self.assertEqual(len(loc), 3)
      self.assertEqual(loc[0], 0)
      self.assertGreaterEqual(loc[1], 0)
      self.assertLessEqual(loc[1], height - patch_size)
      self.assertGreaterEqual(loc[2], 0)
      self.assertLessEqual(loc[2], width - patch_size)

  def test_annotatesImageSizeError(self):
    model = ReshapedModel(Params(image_resize_method = ResizeMethod.NONE))
    img = Image.new('L', (100,200))
    state = model.MakeState(img)
    state[ReshapedLayer.SOURCE] = 'mock source'
    num_patches = 1
    patch_size = 101
    ex = None
    try:
      patches = SamplePatches(model, ReshapedLayer.C1, num_patches, patch_size,
          state)
    except InputSizeError, e:
      ex = e
    self.assertIsNotNone(ex)
    self.assertIsNotNone(ex.source)
    self.assertIsNotNone(ex.layer)

class SamplePatchesFromDataTest(unittest.TestCase):

  def testArrayListInput(self):
    # 4 scales of 100x100 images
    data = [ np.random.random((100, 100)) for _ in range(4) ]
    patches, locs = SamplePatchesFromData(data, patch_width = 10, num_patches = 1)
    self.assertEqual(patches[0].shape, (10, 10))
    self.assertEqual(len(locs[0]), 3)

  def testArrayInput(self):
    data = np.random.random((4, 100, 100))  # 4 scales of 100x100 images
    patches, locs = SamplePatchesFromData(data, patch_width = 10, num_patches = 1)
    self.assertEqual(patches[0].shape, (10, 10))
    self.assertEqual(len(locs[0]), 3)

  def testThreeDimMaps(self):
    data = np.random.random((4, 3, 100, 100))  # 4 scales of 3x100x100 images
    patches, locs = SamplePatchesFromData(data, patch_width = 10, num_patches = 1)
    self.assertEqual(patches[0].shape, (3, 10, 10))
    self.assertEqual(len(locs[0]), 3)

  def testLocationIsAccurate(self):
    data = np.random.random((1, 10, 10))
    patches, locs = SamplePatchesFromData(data, patch_width = 10, num_patches = 1)
    self.assertTrue((locs[0] == (0, 0, 0)).all())

  def testOversampling(self):
    # we should be able to oversample the input
    data = np.random.random((1, 10, 10))
    patches, locs = SamplePatchesFromData(data, patch_width = 10, num_patches = 2)
    self.assertEqual(len(patches), 2)
    self.assertEqual(len(locs), 2)
    # Compare patch data
    self.assertTrue((patches[0] == patches[1]).all())
    # Compare patch locations
    self.assertTrue((locs[0] == locs[1]).all())

  def test_annotatesImageSizeErrorWithScale(self):
    data = np.random.random((1, 10, 10))
    ex = None
    try:
      SamplePatchesFromData(data, patch_width = 11, num_patches = 1)
    except InputSizeError, e:
      ex = e
    self.assertIsNotNone(ex)
    self.assertEqual(ex.scale, 0)

if __name__ == '__main__':
  unittest.main()
