# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import numpy as np
import os
import unittest

from .model import Layer, Model
from .params import Params
from glimpse.models.misc import ImprintS2Prototypes
from glimpse import glab
from glimpse import util

EXAMPLE_IMAGE = os.path.join(glab.GetExampleCorpus(), 'cats', 'Marcus_bed.jpg')
EXAMPLE_IMAGE2 = os.path.join(glab.GetExampleCorpus(), 'dogs',
    '41-27Monate1.JPG')
EXAMPLE_IMAGE_LIST = EXAMPLE_IMAGE, EXAMPLE_IMAGE2
NUM_PROTOTYPES = 10

class TestModel(unittest.TestCase):

  def testSetS2Kernels(self):
    model = Model()
    kernels = [ np.random.random((NUM_PROTOTYPES,) + shape).astype(
        util.ACTIVATION_DTYPE) for shape in model.s2_kernel_shapes ]
    model.s2_kernels = kernels
    self.assertTrue(util.CompareArrayLists(model.s2_kernels, kernels))

  def testSetS2Kernels_badShape(self):
    model = Model()
    kernels = [ np.random.random((NUM_PROTOTYPES, 1) + shape).astype(
        util.ACTIVATION_DTYPE) for shape in model.s2_kernel_shapes ]
    def assign():
      model.s2_kernels = kernels
    self.assertRaises(ValueError, assign)

  def testSetS2Kernels_none(self):
    model = Model()
    kernels = [ np.random.random((NUM_PROTOTYPES,) + shape).astype(
        util.ACTIVATION_DTYPE) for shape in model.s2_kernel_shapes ]
    model.s2_kernels = kernels
    model.s2_kernels = None
    self.assertEqual(model.s2_kernels, None)

  def testBuildLayer(self):
    self._testBuildLayer(params = Params())

  def testBuildLayer_customParams(self):
    params = Params()
    params.num_scales = 8
    params.scale_factor = 2**0.25
    params.s1_num_orientations = 16
    params.s2_kwidth = 7
    self._testBuildLayer(params = params)

  def _testBuildLayer(self, params):
    model = Model(params = params)
    state = model.MakeStateFromFilename(EXAMPLE_IMAGE)
    kernels = [ np.random.random((NUM_PROTOTYPES,) + shape).astype(
        util.ACTIVATION_DTYPE) for shape in model.s2_kernel_shapes ]
    num_kernel_sizes = len(model.s2_kernel_sizes)
    model.s2_kernels = kernels
    lyr = Layer.TopLayer()
    out_state = model.BuildLayer(lyr, state)
    # Check the results
    self.assertNotEqual(out_state, None)
    r = out_state[Layer.RETINA.id]
    s1 = out_state[Layer.S1.id]
    c1 = out_state[Layer.C1.id]
    s2 = out_state[Layer.S2.id]
    c2 = out_state[Layer.C2.id]
    self.assertEqual(r.ndim, 2)
    num_scales = params.num_scales
    self.assertEqual(len(s1), num_scales)
    for s1_scale in s1:
      self.assertEqual(s1_scale.shape[0], params.s1_num_orientations)
    self.assertEqual(len(c1), num_scales)
    for c1_scale in c1:
      self.assertEqual(c1_scale.shape[0], params.s1_num_orientations)
    self.assertEqual(len(s1), num_scales)
    for s2_scale in s2:
      self.assertEqual(s2_scale.shape[0], NUM_PROTOTYPES * num_kernel_sizes)
    self.assertEqual(c2.shape, (NUM_PROTOTYPES * num_kernel_sizes,))

  def testImprintS2Prototypes(self):
    model = Model()
    p = model.params
    input_states = map(model.MakeStateFromFilename, EXAMPLE_IMAGE_LIST)
    prototypes_per_ksize, locations_per_ksize = \
        ImprintS2Prototypes(model, NUM_PROTOTYPES, input_states)
    num_ksizes = len(model.s2_kernel_sizes)
    self.assertEqual(len(prototypes_per_ksize), num_ksizes)
    self.assertEqual(len(locations_per_ksize), num_ksizes)
    for protos, locs, kshape in zip(prototypes_per_ksize, locations_per_ksize,
        model.s2_kernel_shapes):
      self.assertEqual(protos.shape, (NUM_PROTOTYPES,) + kshape)
      self.assertEqual(len(locs), NUM_PROTOTYPES)
      for loc in locs:
        self.assertEqual(len(loc), 4)
        state_idx, scale, y, x = loc
        self.assertTrue(state_idx >= 0 and state_idx < len(input_states))
        self.assertTrue(scale >= 0 and scale < p.num_scales)
