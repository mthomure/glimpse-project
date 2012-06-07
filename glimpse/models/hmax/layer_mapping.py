# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse import backends
from .model import Model
from .params import Params

class LayerSizeMapper(object):
  """Converts spatial extents between model layers.

  These methods compute the image size required to support a model layer (e.g.,
  C2) with at least the given spatial extent (e.g., 3x3 units).

  """

  def __init__(self, backend):
    if backend == None:
      backend = backends.MakeBackend()
    self.backend = backend
    self.params = Params()

  def GetMappingFunction(self, activity_layer, background_layer):
    """Lookup the function that maps activity from one layer to another, where
    layers are identified by strings (e.g., "s1", or "image").

    :param str activity_layer: Identifier of foreground layer.
    :param str background_layer: Identifier of background layer.
    :rtype: callable

    """
    try:
      return getattr(self, "Map%sTo%s" % (activity_layer.name.title(),
          background_layer.name.title()))
    except AttributeError:
      raise ValueError("No function found to map from %s to %s" %
          (activity_layer, background_layer))

  def MapS1ToImage(self, scale, height, width):
    """Compute the size of the input image that resulted in a given S1 map size.

    :param int scale: Scale band index.
    :param int height: S1 map height.
    :param int width: S1 map width.
    :returns: Height and width of input image.
    :rtype: 2-tuple of int

    """
    # All S1 maps for the same scale band are given the same size, which is the
    # size of the smallest map in that band. The smallest map is caused by the
    # largest kernel.
    kw = max( s1.kwidth for s1 in self.params.v1[scale].s1_params )
    scaling = 1
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapC1ToS1(self, scale, height, width):
    """Compute the size of the S1 map that resulted in a given C1 map size.

    :param int scale: Scale band index.
    :param int height: C1 map height.
    :param int width: C1 map width.
    :returns: Height and width of S1 map.
    :rtype: 2-tuple of int

    """
    band_params = self.params.v1[scale]
    kw = band_params.c1_kwidth
    scaling = band_params.c1_scaling
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapC1ToImage(self, height, width):
    """Compute the size of the input image that resulted in a given C1 map size.

    :param int scale: Scale band index.
    :param int height: C1 map height.
    :param int width: C1 map width.
    :returns: Height and width of input image.
    :rtype: 2-tuple of int

    """
    return self.MapS1ToImage(*self.MapC1ToS1(height, width))

  def MapS2ToC1(self, height, width):
    """Compute the size of the C1 map that resulted in a given S2 map size.

    :param int scale: Scale band index.
    :param int height: S2 map height.
    :param int width: S2 map width.
    :returns: Height and width of C1 map.
    :rtype: 2-tuple of int

    """
    # All S2 maps have the same size, which depends on the largest kernel width.
    kw = max(self.params.s2_kwidths)
    scaling = 1
    return self.backend.InputMapShapeForOutput(kw, kw, scaling, height, width)

  def MapS2ToImage(self, height, width):
    """Compute the size of the input image that resulted in a given S2 map size.

    :param int scale: Scale band index.
    :param int height: S2 map height.
    :param int width: S2 map width.
    :returns: Height and width of input image.
    :rtype: 2-tuple of int

    """
    return self.MapC1ToImage(*self.MapS2ToC1(height, width))
