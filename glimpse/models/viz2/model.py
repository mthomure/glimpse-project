# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# Create a 2-part, HMAX-like hierarchy of S+C layers.
# This module implements the "Viz2" model used for the GCNC 2011 experiments.

from glimpse.util import ACTIVATION_DTYPE
from glimpse.util import kernel
from glimpse.models.misc import ImageLayerFromInputArray, SampleC1Patches
import itertools
import numpy as np

class Viz2Model(object):

  def __init__(self, backend, params, s1_kernels = None, s2_kernels = None):
    self.backend = backend
    self.params = params
    if s1_kernels == None:
      s1_kernels = kernel.MakeMultiScaleGaborKernels(
          kwidth = params['s1_kwidth'], num_scales = params['num_scales'],
          num_orientations = params['s1_num_orientations'],
          num_phases = params['s1_num_phases'], shift_orientations = True,
          scale_norm = True)
    else:
      # S1 kernels should have dimensions [scale, orientation, phase, y, x].
      expected_shape = tuple(params[k] for k in ('num_scales',
          's1_num_orientations', 's1_num_phases', 's1_kwidth', 's1_kwidth'))
      assert s1_kernels.shape == expected_shape, \
          "S1 kernels have wrong shape: expected %s but got %s" % \
          (expected_shape, s1_kernels.shape)
      s1_kernels = backend.PrepareArray(s1_kernels)
    if s2_kernels != None:
      # S2 kernels should have dimensions [proto_idx, orientation, y, x].
      ntheta = params['s1_num_orientations']
      kwidth = params['s1_kwidth']
      assert s2_kernels.ndim == 4 and s2_kernels.shape[0] > 0 \
          and s2_kernels.shape[1:] == (ntheta, kwidth, kwidth), \
          "S2 kernels have wrong shape: expected (*, %d, %d, %d) but got %s" % \
          (ntheta, kwidth, kwidth, s2_kernels.shape)
      s2_kernels = backend.PrepareArray(s2_kernels)
    self.s1_kernels = s1_kernels
    self.s2_kernels = s2_kernels

  def BuildImageFromInput(self, input_):
    """Create the initial image layer from some input.
    input_ -- Image or (2-D) array of input data. If array, values should lie in
              the range [0, 1].
    RETURNS (2-D) array containing image layer data
    """
    return ImageLayerFromInputArray(input_, self.backend)

  def BuildRetinaFromImage(self, img):
    if not self.params['retina_enabled']:
      return img
    retina = self.backend.ContrastEnhance(img,
        kwidth = self.params['retina_kwidth'],
        bias = self.params['retina_bias'],
        scaling = 1)
    return retina

  def BuildS1FromRetina(self, retina):
    """Apply S1 processing to some existing retinal layer data.
    retina -- (2-D array) result of retinal layer processing
    RETURNS list of (4-D) S1 activity arrays, with one array per scale
    """
    # Reshape retina to be 3D array
    retina_ = retina.reshape((1,) + retina.shape)
    num_scales = self.params['num_scales']
    s1_kwidth = self.params['s1_kwidth']
    s1_num_orientations = self.params['s1_num_orientations']
    s1_num_phases = self.params['s1_num_phases']
    # Reshape kernel array to be 4-D: scale, index, 1, y, x
    s1_kernels = self.s1_kernels.reshape((num_scales, -1, 1, s1_kwidth,
        s1_kwidth))
    s1s = []
    for scale in range(num_scales):
      ks = s1_kernels[scale]
      s1_ = self.backend.NormRbf(retina_, ks, bias = self.params['s1_bias'],
          beta = self.params['s1_beta'], scaling = self.params['s1_scaling'])
      # Reshape S1 to be 4D array
      s1 = s1_.reshape((s1_num_orientations, s1_num_phases) + s1_.shape[-2:])
      # Pool over phase.
      s1 = s1.max(1)
      s1s.append(s1)
    return s1s

  def BuildC1FromS1(self, s1s):
    num_scales = self.params['num_scales']
    c1s = [ self.backend.LocalMax(s1, kwidth = self.params['c1_kwidth'],
        scaling = self.params['c1_scaling']) for s1 in s1s ]
    # DEBUG: use this to whiten over orientation only
    #~ if self.params['c1_whiten']:
      #~ for c1 in c1s:
        #~ Whiten(c1)
    # DEBUG: use this to whiten over scale AND orientation concurrently. PANN
    # and the old Viz2 model used this.
    if self.params['c1_whiten']:
      c1s = np.array(c1s, ACTIVATION_DTYPE)
      c1_shape = c1s.shape
      c1s = c1s.reshape((-1,) + c1s.shape[-2:])
      Whiten(c1s)
      c1s = c1s.reshape(c1_shape)
    return c1s

  def BuildS2FromC1(self, c1s):
    CheckPrototypes(self.s2_kernels)
    num_scales = self.params['num_scales']
    s2s = []
    for scale in range(num_scales):
      c1 = c1s[scale]
      s2 = self.backend.NormRbf(c1, self.s2_kernels,
          bias = self.params['s2_bias'], beta = self.params['s2_beta'],
          scaling = self.params['s2_scaling'])
      s2s.append(s2)
    return s2s

  def BuildC2FromS2(self, s2s):
    c2s = map(self.backend.GlobalMax, s2s)
    return c2s

  def BuildItFromC2(self, c2s):
    it = np.array(c2s).max(0)
    return it

  def BuildLayers(self, data, input_layer, output_layer):
    """Process the input data, building up to the given output layer.
    data -- input array
    input_layer -- layer ID of input array
    output_layer -- layer ID of output array
    """
    start = ALL_LAYERS.index(input_layer)
    stop = ALL_LAYERS.index(output_layer)
    layers = ALL_LAYERS[start : stop + 1]
    results = dict()
    for idx in range(start, stop + 1):
      builder = ALL_BUILDERS[idx]
      data = builder(self, data)
      layer_name = ALL_LAYERS[idx]
      results[layer_name] = data
    return results

  def ImprintPrototypes(self, input_, num_prototypes, normalize = True):
    """Compute C1 activity maps and sample patches from random locations.
    input_ -- (Image or 2-D array) unprocessed image data
    num_prototype -- (positive int) number of prototypes to imprint
    normalize -- (bool) whether to scale each prototype to have unit length
    RETURNS list of prototypes, and list of corresponding locations.
    """
    results = self.BuildLayers(input_, LAYER_IMAGE, LAYER_C1)
    c1s = results[LAYER_C1]
    proto_it = SampleC1Patches(c1s, self.params['s2_kwidth'])
    protos = list(itertools.islice(proto_it, num_prototypes))
    for proto, loc in protos:
      proto /= np.linalg.norm(proto)
    return zip(*protos)

# Identifiers for layers that can be computed
LAYER_IMAGE = 'i'
LAYER_RETINA = 'r'
LAYER_S1 = 's1'
LAYER_C1 = 'c1'
LAYER_S2 = 's2'
LAYER_C2 = 'c2'
LAYER_IT = 'it'
# The set of all layers in this model, in order of processing.
ALL_LAYERS = (LAYER_IMAGE, LAYER_RETINA, LAYER_S1, LAYER_C1, LAYER_S2, LAYER_C2,
    LAYER_IT)
ALL_BUILDERS = (Viz2Model.BuildImageFromInput, Viz2Model.BuildRetinaFromImage,
    Viz2Model.BuildS1FromRetina, Viz2Model.BuildC1FromS1,
    Viz2Model.BuildS2FromC1, Viz2Model.BuildC2FromS2, Viz2Model.BuildItFromC2)

def Whiten(data):
  data -= data.mean(0)
  norms = np.sqrt((data**2).sum(0))
  norms[ norms < 1 ] = 1
  data /= norms
  return data

def CheckPrototypes(prototypes):
  assert prototypes != None
  if len(prototypes.shape) == 3:
    prototypes = prototypes.reshape((1,) + prototypes.shape)
  assert np.allclose(np.array(map(np.linalg.norm, prototypes)), 1), \
      "Internal error: S2 prototypes are not normalized"
  assert not np.isnan(prototypes).any(), \
      "Internal error: found NaN in imprinted prototype."
