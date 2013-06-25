from glimpse.util.gtest import *
from .param import *

class TestCase(unittest.TestCase):

  def testResizeMethod_badInitValue(self):
    with self.assertRaises(ValueError):
      ResizeMethod("bad-value-to-resize-method")

  def testParams_eq(self):
    p1 = Params(image_resize_aspect_ratio = 0.1)
    p2 = Params(image_resize_aspect_ratio = 0.1)
    self.assertEqual(p1, p2)

  def testParams_notEq(self):
    p1 = Params(image_resize_aspect_ratio = 0.1)
    p2 = Params(image_resize_aspect_ratio = 0.5)
    self.assertNotEqual(p1, p2)

if __name__ == '__main__':
  unittest.main()
