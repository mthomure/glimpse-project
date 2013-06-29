from .misc import *
from . import misc
from .experiment import ExperimentData
from glimpse.models.ml import Model, State, Layer
from glimpse.util.gtest import *

class Tests(unittest.TestCase):

  def testGetActivity_strImage(self):
    image_name = 'fake-image-name'
    layer = 'image'
    layer_data = 'IMAGE-DATA'
    exp = ExperimentData()
    exp.extractor.model = Model()
    f = RecordedFunctionCall(State({layer:layer_data}))
    with MonkeyPatch(misc, 'BuildLayer', f):
      data = misc.GetActivity(exp, image_name, layer)
    self.assertEqual(data, layer_data)
    self.assertTrue(f.called)
    self.assertEqual(f.args[1], layer)
    self.assertEqual(f.args[2][Layer.SOURCE].image_path, image_name)

  def testGetActivity_intImage_withActivity_withLayer(self):
    image_name = 0
    layer = 'image'
    layer_data = 'IMAGE-DATA'
    exp = ExperimentData()
    exp.extractor.activation = [State({layer:layer_data})]
    f = RecordedFunctionCall(State({layer:None}))
    with MonkeyPatch(misc, 'BuildLayer', f):
      data = misc.GetActivity(exp, image_name, layer)
    self.assertEqual(data, layer_data)
    self.assertFalse(f.called)

  def testGetActivity_intImage_withActivity_noLayer(self):
    image_name = 0
    image_path = 'fake-image-path'
    layer = 'image'
    layer_data = 'IMAGE-DATA'
    exp = ExperimentData()
    exp.extractor.model = Model()
    exp.corpus.paths = [image_path]
    exp.extractor.activation = [State()]
    f = RecordedFunctionCall(State({layer:layer_data}))
    with MonkeyPatch(misc, 'BuildLayer', f):
      data = misc.GetActivity(exp, image_name, layer)
    self.assertEqual(data, layer_data)
    self.assertTrue(f.called)
    self.assertEqual(f.args[1], layer)

  def testGetActivity_intImage_noActivity(self):
    image_name = 0
    image_path = 'fake-image-path'
    layer = 'image'
    layer_data = 'IMAGE-DATA'
    exp = ExperimentData()
    exp.corpus.paths = [image_path]
    exp.extractor.model = Model()
    f = RecordedFunctionCall(State({layer:layer_data}))
    with MonkeyPatch(misc, 'BuildLayer', f):
      data = misc.GetActivity(exp, image_name, layer)
    self.assertEqual(data, layer_data)
    self.assertTrue(f.called)
    self.assertEqual(f.args[1], layer)
    self.assertEqual(f.args[2][Layer.SOURCE].image_path, image_path)
