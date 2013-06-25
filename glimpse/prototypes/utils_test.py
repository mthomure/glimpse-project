import unittest

from .utils import *
from glimpse.backends import ACTIVATION_DTYPE
from glimpse.models.ml import Model, Layer, Params
from glimpse.models.base import LayerSpec, BuildLayer
from glimpse.models.base.layer import layer_builder
from glimpse.pools import SinglecorePool
from glimpse.util.garray import toimage

class ReshapedLayer(Layer):

  RESHAPED_IMAGE = LayerSpec("r", "reshape-image", [Layer.IMAGE])

class ReshapedModel(Model):

  LayerClass = ReshapedLayer

  @layer_builder(ReshapedLayer.RESHAPED_IMAGE)
  def ReshapeImage(self, img):
    return img.reshape((1,) + img.shape)

class SamplesPatchesFromImagesTests(unittest.TestCase):

  def _test(self, num_kernels, num_images, normalize):
    p = Params(num_scales = 3)
    model = ReshapedModel(p)
    kernel_sizes = (5, 7, 11)
    states = [ model.MakeState(toimage(np.random.random((100,100))))
        for _ in range(num_images) ]
    kernels, locs = SamplePatchesFromImages(model, ReshapedLayer.RESHAPED_IMAGE,
        kernel_sizes, num_kernels, states, SinglecorePool(),
        normalize = normalize)
    self.assertEqual(len(kernels), len(kernel_sizes))
    self.assertEqual(len(locs), len(kernel_sizes))
    for idx, ksize in enumerate(kernel_sizes):
      self.assertEqual(len(kernels[idx]), num_kernels)
      self.assertEqual(len(locs[idx]), num_kernels)
      for k, l in zip(kernels[idx], locs[idx]):
        self.assertEqual(k.ndim, 2)
        self.assertEqual(len(l), 4)
        if normalize:
          self.assertAlmostEqual(np.linalg.norm(k), 1,
              places = 5)  # don't be too strict with floating-point comparison

  def testMultipleImagesAndMultipleKernels(self):
    self._test(16, 5, False)

  def testSingleImage(self):
    self._test(10, 1, False)

  def testSingleKernel(self):
    self._test(1, 2, False)

  def testSameNumberOfKernelsAndImages(self):
    self._test(3, 3, False)

  def testNormed(self):
    self._test(20, 10, True)

  def testLocationsAreCorrect(self):
    # Imprinted patches must match the data at the given image locations.
    p = Params(num_scales = 3)
    model = ReshapedModel(p)
    num_images = 8
    num_patches = 7
    states = [ model.MakeState(np.random.random((100,100)).astype(ACTIVATION_DTYPE))
        for _ in range(num_images) ]
    pwidth = 11
    patches,locs = SamplePatchesFromImages(model, Layer.IMAGE, [pwidth],
        num_patches, states)
    patches = patches[0]  # using single patch width
    locs = locs[0]
    for patch,loc in zip(patches,locs):
      idx,_,y,x = loc  # ignoring scale band, since image is not multi-scale
      img_st = BuildLayer(model, Layer.IMAGE, model.MakeState(states[idx]),
          save_all=False)
      img = img_st['i']
      img_patch = img[y:y+pwidth, x:x+pwidth]
      self.assertTrue(np.all(img_patch == patch))

if __name__ == '__main__':
  unittest.main()
