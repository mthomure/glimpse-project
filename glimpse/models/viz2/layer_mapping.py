# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from model import Layer

class CoordinateMapper(object):
  """Given an offset expressed in the coordinates of one layer, these methods
  convert that value to the equivalent offset in the coordinates of a lower
  layer. Note that these methods assume that all kernels are square."""

  def __init__(self, params, center = True):
    self.params = params
    self.center = center

  def GetMappingFunction(self, activity_layer, background_layer):
    """Lookup the function that maps activity from one layer to another, where
    layers are identified by strings (e.g., "s1", or "image")."""
    try:
      return getattr(self, "Map%sTo%s" % (activity_layer.name.title(),
          background_layer.name.title()))
    except AttributeError:
      raise ValueError("No function found to map from %s to %s" %
          (activity_layer, background_layer))

  def MapRetinaToImage(self, x):
    if self.params.retina_enabled and self.center:
      r_kw = self.params.retina_kwidth
      x += r_kw / 2
    return int(x)

  def MapS1ToRetina(self, x):
    s1_s = self.params.s1_scaling
    x = x * s1_s
    if self.center:
      s1_kw = self.params.s1_kwidth
      x += s1_kw / 2
    return int(x)

  def MapS1ToImage(self, x):
    return self.MapRetinaToImage(self.MapS1ToRetina(x))

  def MapC1ToS1(self, x):
    c1_s = self.params.c1_scaling
    x = x * c1_s
    if self.center:
      c1_kw = self.params.c1_kwidth
      x += c1_kw / 2
    return int(x)

  def MapC1ToRetina(self, x):
    return self.MapS1ToRetina(self.MapC1ToS1(x))

  def MapC1ToImage(self, x):
    return self.MapRetinaToImage(self.MapC1ToRetina(x))

  def MapS2ToC1(self, x):
    s2_s = self.params.s2_scaling
    x = x * s2_s
    if self.center:
      s2_kw = self.params.s2_kwidth
      x += s2_kw / 2
    return int(x)

  def MapS2ToImage(self, x):
    return self.MapC1ToImage(self.MapS2ToC1(x))

  def MapC2ToS2(self, x):
    c2_s = self.params.c2_scaling
    x = x * c2_s
    if self.center:
      c2_kw = self.params.c2_kwidth
      x += c2_kw / 2
    return int(x)

  def MapC2ToImage(self, x):
    return self.MapS2ToImage(self.MapC2ToS2(x))

class LayerSizeMapper(object):
  """These methods compute the image size required to support a model layer
  (e.g., C2) with at least the given spatial extent (e.g., 3x3 units)."""

  def __init__(self, backend, params):
    self.backend = backend
    self.params = params

  def GetMappingFunction(self, activity_layer, background_layer):
    """Lookup the function that maps activity from one layer to another, where
    layers are identified by strings (e.g., "s1", or "image")."""
    try:
      return getattr(self, "Map%sTo%s" % (activity_layer.name.title(),
          background_layer.name.title()))
    except AttributeError:
      raise ValueError("No function found to map from %s to %s" %
          (activity_layer, background_layer))

  def MapRetinaToImage(self, height, width):
    kw = self.params.retina_kwidth
    return self.backend.InputMapShapeForOutput(kw, kw,
        1, # scaling
        height, width)

  def MapS1ToRetina(self, height, width):
    kw = self.params.s1_kwidth
    scaling = self.params.s1_scaling
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapS1ToImage(self, height, width):
    return self.MapRetinaToImage(*self.MapS1ToRetina(height, width))

  def MapC1ToS1(self, height, width):
    kw = self.params.c1_kwidth
    scaling = self.params.c1_scaling
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapC1ToImage(self, height, width):
    return self.MapS1ToImage(*self.MapC1ToS1(height, width))

  def MapS2ToC1(self, height, width):
    kw = self.params.s2_kwidth
    scaling = self.params.s2_scaling
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapS2ToImage(self, height, width):
    return self.MapC1ToImage(*self.MapS2ToC1(height, width))

  def MapC2ToS2(self, height, width):
    kw = self.params.c2_kwidth
    scaling = self.params.c2_scaling
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapC2ToImage(self, height, width):
    return self.MapS2ToImage(*self.MapC2ToS2(height, width))

class RegionMapper(object):
  """The methods of this object re-express a slice --- given in coordinates
  of one layer --- in terms of coordinates of some lower-level layer. Note
  that these methods assume that all kernels are square."""

  def __init__(self, params):
    self.params = params
    self.cmapper = CoordinateMapper(params, center = False)

  def GetMappingFunction(self, activity_layer, background_layer):
    """Lookup the function that maps activity from one layer to another, where
    layers are identified by strings (e.g., "s1", or "image").
    activity_layer -- identifier of foreground layer
    background_layer -- identifier of background layer
    """
    try:
      return getattr(self, "Map%sTo%s" % (activity_layer.name.title(),
          background_layer.name.title()))
    except AttributeError:
      raise ValueError("No function found to map from %s to %s" %
          (activity_layer, background_layer))

  def MapRetinaToImage(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapRetinaToImage
    if self.params.retina_enabled:
      offset = self.params.retina_kwidth
    else:
      offset = 1  # retina width is 1 unit
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapS1ToRetina(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapS1ToRetina
    offset = self.params.s1_kwidth
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapS1ToImage(self, x):
    return self.MapRetinaToImage(self.MapS1ToRetina(x))

  def MapC1ToS1(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapC1ToS1
    offset = self.params.c1_kwidth
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapC1ToRetina(self, x):
    return self.MapS1ToRetina(self.MapC1ToS1(x))

  def MapC1ToImage(self, x):
    """Map C1 region to image coordinates.
    x -- (slice) region in C1 coordinates
    """
    return self.MapRetinaToImage(self.MapC1ToRetina(x))

  def MapS2ToC1(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapS2ToC1
    offset = self.params.s2_kwidth
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapS2ToImage(self, x):
    return self.MapC1ToImage(self.MapS2ToC1(x))

  def MapC2ToS2(self, x):
    assert x.start < x.stop, "Slice must be non-empty"
    f = self.cmapper.MapC2ToS2
    offset = self.params.c2_kwidth
    return slice(f(x.start), f(x.stop - 1) + offset)

  def MapC2ToImage(self, x):
    return self.MapS2ToImage(self.MapC2ToS2(x))
