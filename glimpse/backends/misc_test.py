import importlib
import numpy as np
import os
import unittest

from .misc import *
from .base_backend import BaseBackend
from .scipy_backend import ScipyBackend
from glimpse.util.gtest import *
from glimpse.util.garray import toimage

# We assume DEFAULT_BACKEND is "base", and that both BaseBackend and
# ScipyBackend are available.

class TestMisc(unittest.TestCase):

  def testMakeBackend_default(self):
    if 'GLIMPSE_BACKEND' in os.environ:
      del os.environ['GLIMPSE_BACKEND']
    self.assertIsInstance(MakeBackend(), BaseBackend)

  def testMakeBackend_withEnvVar(self):
    os.environ['GLIMPSE_BACKEND'] = "scipy"
    self.assertIsInstance(MakeBackend(), ScipyBackend)

  def testMakeBackend_withArg(self):
    os.environ['GLIMPSE_BACKEND'] = "base"  # argument overrides env var
    self.assertIsInstance(MakeBackend("scipy"), ScipyBackend)

class TestInputSource(unittest.TestCase):

  def testCreateImage(self):
    with TempFile(suffix = '.jpg') as path:
      toimage(np.random.random((256, 128)) * 255).save(path)
      source = InputSource(path)
      img = source.CreateImage()
      self.assertIsNotNone(img)
      self.assertEqual(img.size, (128, 256))
      self.assertEqual(img.fp.name, path)

  def testCreateImage_throwsLoadException(self):
    source = InputSource('/invalid-file-name.jpg')
    self.assertRaises(InputLoadError, source.CreateImage)

  def testEqual(self):
    path = '/example/path/to/image.jpg'
    source1 = InputSource(path)
    source2 = InputSource(path)
    self.assertEqual(source1, source2)

if __name__ == '__main__':
  unittest.main()
