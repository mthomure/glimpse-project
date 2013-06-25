import unittest
import Image
import numpy as np

from .model import Model, Layer, State, layer_builder
from .layer import LayerSpec
from .param import *
from glimpse.backends import ACTIVATION_DTYPE
from glimpse.util import dataflow

class Model1(object): pass
class S1(State):
  ModelClass = Model1

class Model2(object): pass
class S2(State):
  ModelClass = Model2

l1 = LayerSpec("l1", "layer-1")
l2 = LayerSpec("l2", "layer-2", [l1])

l1_state = 'l1-state'
l2_state = 'l2-state'

def FillState(s):
  s[l1] = l1_state
  s[l2] = l2_state
  return s

class ReshapedLayer(Layer):

  RESHAPED_IMAGE = LayerSpec("r", "reshape-image", [Layer.IMAGE])

class ReshapedModel(Model):

  LayerClass = ReshapedLayer

  @layer_builder(ReshapedLayer.RESHAPED_IMAGE)
  def ReshapeImage(self, img):
    return img.reshape((1,) + img.shape)

class S2Layer(Layer):

  C1 = LayerSpec("c1", "C1", [Layer.IMAGE])

class S2Model(Model):

  LayerClass = S2Layer

  @property
  def s2_kernel_widths(self):
    return (2, 3)

  @layer_builder(S2Layer.C1)
  def BuildC1(self, data):
    # Return the array with one scale and one band.
    return data.reshape((1, 1) + data.shape)

class StateTest(unittest.TestCase):

  def testEq(self):
    # States with same contents but different types are not equal.
    s1 = FillState(S1())
    s1b = FillState(S1())
    s2 = FillState(S2())
    self.assertEqual(s1, s1b)
    self.assertNotEqual(s1, s2)

class ModelTest(unittest.TestCase):

  def testMakeState_filename(self):
    m = Model()
    path = "/test/path"
    s = m.MakeState(path)
    self.assertSequenceEqual(s.keys(), (Layer.SOURCE.id_,))
    self.assertEqual(s[Layer.SOURCE].image_path, path)

  def testMakeState_image(self):
    params = Params()
    params.image_resize_method = "none"
    m = Model(params)
    img = Image.new('L', (100,100))
    s = m.MakeState(img)
    self.assertSequenceEqual(s.keys(), (Layer.IMAGE.id_,))
    self.assertSequenceEqual(s[Layer.IMAGE].shape, (100,100))

  def testMakeState_array(self):
    m = Model()
    img = np.zeros((100,100), dtype = ACTIVATION_DTYPE)
    s = m.MakeState(img)
    self.assertSequenceEqual(s.keys(), (Layer.IMAGE.id_,))
    self.assertTrue(np.all(s[Layer.IMAGE] == img))

  def testBuildNonImageLayer(self):
    params = Params()
    params.image_resize_method = "none"
    m = ReshapedModel(params)
    img = Image.new('L', (100,100))
    s = dataflow.BuildNode(m, ReshapedLayer.RESHAPED_IMAGE, m.MakeState(img))
    self.assertIn(ReshapedLayer.RESHAPED_IMAGE, s)
    self.assertSequenceEqual(s[ReshapedLayer.RESHAPED_IMAGE].shape,
        (1, 100,100))

if __name__ == '__main__':
  unittest.main()
