# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import pprint

from glimpse.util import traits

class ResizeMethod(traits.Enum):
  """A trait type describing how to resize an input image."""

  NONE = "none"
  SCALE_SHORT_EDGE = "scale short edge"
  SCALE_LONG_EDGE = "scale long edge"
  SCALE_WIDTH = "scale width"
  SCALE_HEIGHT = "scale height"
  SCALE_AND_CROP = "scale and crop"

  def __init__(self, value, **metadata):
    values = (self.NONE, self.SCALE_SHORT_EDGE, self.SCALE_LONG_EDGE,
        self.SCALE_WIDTH, self.SCALE_HEIGHT, self.SCALE_AND_CROP)
    if value not in values:
      raise ValueError("Expected value (%s) to be one of: %s" % (value,
          ", ".join(values)))
    super(ResizeMethod, self).__init__(value, values, **metadata)

class Params(traits.HasStrictTraits):
  """Parameter container for the :class:`.model.Model`."""

  # When the method is SCALE_AND_CROP, use the length parameter to specify the
  # output image width, and the aspect_ratio parameter to specify the (relative)
  # output image height.
  image_resize_method = ResizeMethod(ResizeMethod.SCALE_SHORT_EDGE,
      label = "Image Resize Method", desc = "method for resizing input images")

  image_resize_length = traits.Range(0, value = 220, exclude_low = True,
      label = "Image Resize Length", desc = "Length of resized image")

  image_resize_aspect_ratio = traits.Range(0., value = 1., exclude_low = True,
      label = 'Image Resize Aspect Ratio',
      desc = 'Aspect ratio of resized image')

  def __init__(self, **kw):
    super(Params, self).__init__()
    for k, v in kw.items():
      setattr(self, k, v)

  def __str__(self):
    # Get list of all traits.
    keys = self.traits().keys()
    # Remove special entries of the HasTraits object.
    keys = filter((lambda t: not t.startswith("trait_")), keys)
    # Display traits in alphabetical order.
    keys = sorted(keys)
    # Format set of traits as a string.
    return "%s(\n  %s\n)" % (type(self).__name__, ("\n  ".join("%s = %s," % (k,
        pprint.pformat(getattr(self, k))) for k in keys)))

  def __eq__(self, other):
    if type(self) != type(other):
      return False
    # Get list of all traits.
    keys = self.traits().keys()
    # Remove special entries of the HasTraits object.
    keys = filter((lambda t: not t.startswith("trait_")), keys)
    for k in keys:
      if getattr(self, k) != getattr(other, k):
        return False
    return True

  __repr__ = __str__
