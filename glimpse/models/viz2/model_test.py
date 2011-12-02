# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import glimpse
from glimpse.models.misc import DependencyError
import Image
from model import Model, State
import numpy as np
from params import Params
import unittest

class TestModel(unittest.TestCase):

  ####### UTILITY FUNCTIONS ########

  NUM_PROTOTYPES = 10
  NUM_SCALES = 4
  NUM_ORIENTATIONS = 8

  def MakeModel(self, layer):
    params = Params()
    params.num_scales = self.NUM_SCALES
    params.s1_num_orientations = self.NUM_ORIENTATIONS
    model = Model(params = params)
    L = model.Layer
    if layer in (L.S2, L.C2, L.IT):
      # Make uniform-random S2 kernels
      kernel_shape = (self.NUM_PROTOTYPES,) + model.s2_kernel_shape
      kernels = np.random.uniform(0, 1, size = kernel_shape)
      for k in kernels:
        k /= np.linalg.norm(k)
      model.s2_kernels = kernels
    return model

  def BuildLayerFromState(self, model, layer, state, result_type = None):
    state = model.BuildLayer(layer, state)
    result = state.get(layer.id, None)
    self.assertNotEqual(result, None)
    if result_type != None:
      self.assertTrue(isinstance(result, result_type))
    return result

  def BuildLayerFromPath(self, layer, result_type = None):
    # Load an example image
    model = self.MakeModel(layer)
    path = glimpse.GetExampleImagePaths()[0]
    state = model.MakeStateFromFilename(path)
    return self.BuildLayerFromState(model, layer, state, result_type)

  def BuildLayerFromInMemoryImage(self, layer, result_type = None):
    model = self.MakeModel(layer)
    path = glimpse.GetExampleImagePaths()[0]
    img = Image.open(path)
    state = model.MakeStateFromImage(img)
    return self.BuildLayerFromState(model, layer, state, result_type)

  ###### TESTS ########

  def testEmptyStateFails(self):
    model = Model()
    state = State()
    def f():
      return model.BuildLayer(Model.Layer.IMAGE, state)
    self.assertRaises(DependencyError, f)

  def testBuildImageLayerFromPath(self):
    layer = Model.Layer.IMAGE
    model = self.MakeModel(layer)
    path = glimpse.GetExampleImagePaths()[0]
    state = model.MakeStateFromFilename(path)
    result = self.BuildLayerFromState(model, layer, state,
        result_type = np.ndarray)
    img = Image.open(path)
    self.assertEqual(result.shape, img.size[::-1])

  def testBuildRetinaFromPath(self):
    self.BuildLayerFromPath(Model.Layer.RETINA, np.ndarray)

  def testBuildRetinaFromInMemoryImage(self):
    self.BuildLayerFromInMemoryImage(Model.Layer.RETINA, np.ndarray)

  def testBuildS1(self):
    result = self.BuildLayerFromPath(Model.Layer.S1, np.ndarray)
    self.assertEqual(result.ndim, 4)
    self.assertEqual(result.shape[:-2], (self.NUM_SCALES,
        self.NUM_ORIENTATIONS))

  def testBuildC1(self):
    result = self.BuildLayerFromPath(Model.Layer.C1, np.ndarray)
    self.assertEqual(result.ndim, 4)
    self.assertEqual(result.shape[:-2], (self.NUM_SCALES,
        self.NUM_ORIENTATIONS))

  def testBuildS2(self):
    result = self.BuildLayerFromPath(Model.Layer.S2, np.ndarray)
    self.assertEqual(result.ndim, 4)
    self.assertEqual(result.shape[:-2], (self.NUM_SCALES, self.NUM_PROTOTYPES))

  def testBuildC2(self):
    result = self.BuildLayerFromPath(Model.Layer.C2, np.ndarray)
    self.assertEqual(result.ndim, 2)
    self.assertEqual(result.shape, (self.NUM_SCALES, self.NUM_PROTOTYPES))

  def testBuildIt(self):
    result = self.BuildLayerFromPath(Model.Layer.IT, np.ndarray)
    self.assertEqual(result.ndim, 1)
    self.assertEqual(result.shape, (self.NUM_PROTOTYPES,))

  def testSampleC1Patches(self):
    num_patches = 10
    model = self.MakeModel(Model.Layer.C1)
    path = glimpse.GetExampleImagePaths()[0]
    state = model.MakeStateFromFilename(path)
    result = model.SampleC1Patches(num_patches, state, normalize = True)
    self.assertEqual(len(result), num_patches)
    patches, locations = zip(*result)
    locations = np.array(locations)
    norms = np.array(map(np.linalg.norm, patches))
    self.assertTrue(np.allclose(norms, 1))
    self.assertEqual(locations.ndim, 2)
    self.assertTrue(np.all(locations >= 0))

if __name__ == '__main__':
  unittest.main()
