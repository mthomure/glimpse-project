# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import numpy as np
import os

from glimpse.models.base.misc import BuildLayer
from .misc import Whiten
from .model import Layer, Model, State
from .param import Params, SLayerOp
from glimpse.util.garray import CompareArrayLists
from glimpse.util.garray import toimage
from glimpse.backends.base_backend import BaseBackend, BackendError, \
    ACTIVATION_DTYPE
from glimpse.util.gtest import *

class BrokenBackend(BaseBackend):

  def __init__(self, delay = 0):
    super(BrokenBackend, self).__init__()
    self.delay = delay

  def ContrastEnhance(self, *args, **kw):
    if self.delay == 0:
      raise BackendError
    self.delay -= 1
    return super(BrokenBackend, self).ContrastEnhance(*args, **kw)

  def DotProduct(self, *args, **kw):
    if self.delay == 0:
      raise BackendError
    self.delay -= 1
    return super(BrokenBackend, self).DotProduct(*args, **kw)

  def NormDotProduct(self, *args, **kw):
    if self.delay == 0:
      raise BackendError
    self.delay -= 1
    return super(BrokenBackend, self).DotProduct(*args, **kw)

  def LocalMax(self, *args, **kw):
    if self.delay == 0:
      raise BackendError
    self.delay -= 1
    return super(BrokenBackend, self).LocalMax(*args, **kw)

  def Rbf(self, *args, **kw):
    if self.delay == 0:
      raise BackendError
    self.delay -= 1
    return super(BrokenBackend, self).Rbf(*args, **kw)

  def GlobalMax(self, *args, **kw):
    if self.delay == 0:
      raise BackendError
    self.delay -= 1
    return super(BrokenBackend, self).GlobalMax(*args, **kw)

class S1KernelTest(unittest.TestCase):

  def testGet_none(self):
    p = Params()
    model = Model(p)
    kernels = model.s1_kernels
    self.assertEqual(kernels.ndim, 4)
    self.assertSequenceEqual(kernels.shape, p.s1_kernel_shape)

  def testGet_notNone(self):
    model = Model()
    model._s1_kernels = 2
    self.assertEqual(model.s1_kernels, 2)

  def testSet_notNormed(self):
    p = Params(s1_operation = SLayerOp.DOT_PRODUCT)
    model = Model(p)
    kernels = np.random.random(p.s1_kernel_shape).astype(ACTIVATION_DTYPE)
    model.s1_kernels = kernels

  def testSet_normed(self):
    p = Params(s1_operation = SLayerOp.NORM_DOT_PRODUCT)
    model = Model(p)
    kernels = np.random.random(p.s1_kernel_shape).astype(ACTIVATION_DTYPE)
    for k in kernels.reshape(-1, p.s1_kwidth, p.s1_kwidth):
      k /= np.linalg.norm(k)
    model.s1_kernels = kernels

  def testSet_none(self):
    model = Model()
    model._s1_kernels = 2
    model.s1_kernels = None
    self.assertIsNone(model._s1_kernels)

  def testSet_errorOnBadShape(self):
    p = Params(s1_operation = SLayerOp.DOT_PRODUCT)
    model = Model(p)
    kernels = np.random.random((2,) + p.s1_kernel_shape).astype(
        ACTIVATION_DTYPE)
    with self.assertRaises(ValueError):
      model.s1_kernels = kernels

  def testSet_errorOnNotNormed(self):
    p = Params(s1_operation = SLayerOp.NORM_DOT_PRODUCT)
    model = Model(p)
    kernels = np.random.random(p.s1_kernel_shape).astype(ACTIVATION_DTYPE)
    kernels.flat[0] = 100  # ensure norm is not accidentally close to one
    with self.assertRaises(ValueError):
      model.s1_kernels = kernels

  def testSet_errorOnNan(self):
    p = Params(s1_operation = SLayerOp.DOT_PRODUCT)
    model = Model(p)
    kernels = np.random.random(p.s1_kernel_shape).astype(ACTIVATION_DTYPE)
    kernels.flat[0] = np.nan
    with self.assertRaises(ValueError):
      model.s1_kernels = kernels


class S2KernelTest(unittest.TestCase):

  def testGet_none(self):
    model = Model()
    self.assertIsNone(model.s2_kernels)

  def testGet_notNone(self):
    model = Model()
    model._s2_kernels = 2
    self.assertEqual(model.s2_kernels, 2)

  def testSet(self):
    p = Params()
    model = Model(p)
    kernels = [ np.random.random((10,) + shape).astype(ACTIVATION_DTYPE)
        for shape in p.s2_kernel_shapes ]
    model.s2_kernels = kernels
    self.assertTrue(CompareArrayLists(model.s2_kernels, kernels))

  def testSet_errorOnBadShape(self):
    p = Params()
    model = Model(p)
    kernels = [ np.random.random((10,1) + shape).astype(ACTIVATION_DTYPE)
        for shape in p.s2_kernel_shapes ]
    with self.assertRaises(ValueError):
      model.s2_kernels = kernels

  def testSet_errorOnNan(self):
    p = Params()
    model = Model(p)
    kernels = [ np.random.random((10,) + shape).astype(ACTIVATION_DTYPE)
        for shape in p.s2_kernel_shapes ]
    kernels[0].flat[0] = np.nan
    with self.assertRaises(ValueError):
      model.s2_kernels = kernels

  def testSet_none(self):
    p = Params()
    model = Model(p)
    kernels = [ np.random.random((10,) + shape)
        for shape in p.s2_kernel_shapes ]
    model.s2_kernels = kernels
    model.s2_kernels = None
    self.assertEqual(model.s2_kernels, None)


class RetinaTest(unittest.TestCase):

  def testSuccess(self):
    model = Model(Params(retina_enabled = True))
    img = np.random.random((100,100)).astype(ACTIVATION_DTYPE)
    result = model.BuildRetina(img)
    self.assertEqual(result.ndim, 2)

  def testSuccess_disabled(self):
    model = Model(Params(retina_enabled = False))
    img = np.random.random((100,100)).astype(ACTIVATION_DTYPE)
    result = model.BuildRetina(img)
    self.assertTrue(np.all(img == result))

  def testAnnotatesError(self):
    model = Model(backend = BrokenBackend(),
        params = Params(retina_enabled = True))
    img = np.random.random((100,100)).astype(ACTIVATION_DTYPE)
    ex = None
    try:
      model.BuildRetina(img)
    except BackendError, e:
      ex = e
    self.assertIsNotNone(ex)
    self.assertEqual(ex.layer, Layer.RETINA)


class S1Test(unittest.TestCase):

  def testSuccess_dotProduct(self):
    p = Params(s1_operation = SLayerOp.DOT_PRODUCT, num_scales = 2)
    model = Model(p)
    retina = np.random.random((100,100)).astype(ACTIVATION_DTYPE)
    result = model.BuildS1(retina)
    self.assertEqual(len(result), p.num_scales)
    self.assertTrue(all(r.ndim == 3 for r in result))

  def testSuccess_normDotProduct(self):
    p = Params(s1_operation = SLayerOp.NORM_DOT_PRODUCT, num_scales = 2)
    model = Model(p)
    retina = np.random.random((100,100)).astype(ACTIVATION_DTYPE)
    result = model.BuildS1(retina)
    self.assertEqual(len(result), p.num_scales)
    self.assertTrue(all(r.ndim == 3 for r in result))

  def testAnnotatesError(self):
    scale_for_error = 2
    model = Model(BrokenBackend(delay = scale_for_error))
    retina = np.random.random((100,100)).astype(ACTIVATION_DTYPE)
    ex = None
    try:
      model.BuildS1(retina)
    except BackendError, e:
      ex = e
    self.assertIsNotNone(ex)
    self.assertEqual(ex.layer, Layer.S1)
    self.assertEqual(ex.scale, scale_for_error)


class C1Test(unittest.TestCase):

  def testSuccess(self):
    p = Params(num_scales = 2)
    model = Model(p)
    s1 = [ np.random.random((1, 100, 100)).astype(ACTIVATION_DTYPE)
        for _ in range(p.num_scales) ]
    result = model.BuildC1(s1)
    self.assertEqual(len(result), p.num_scales)
    self.assertTrue(all(x.ndim == 3 for x in result))

  def testAnnotatesError(self):
    scale_for_error = 1
    p = Params(num_scales = 2)
    model = Model(BrokenBackend(scale_for_error), p)
    ex = None
    s1 = [ np.random.random((1, 100, 100)).astype(ACTIVATION_DTYPE)
        for _ in range(p.num_scales) ]
    try:
      model.BuildC1(s1)
    except BackendError, e:
      ex = e
    self.assertIsNotNone(ex)
    self.assertEqual(ex.layer, Layer.C1)
    self.assertEqual(ex.scale, scale_for_error)

  def testWithWhiten(self):
    whiten_replacement = RecordedFunctionCall()
    result = None
    with MonkeyPatch('glimpse.models.ml.model', 'Whiten', whiten_replacement):
      p = Params(c1_whiten = True, num_scales = 2)
      model = Model(p)
      s1 = [ np.random.random((1, 100, 100)).astype(ACTIVATION_DTYPE)
          for _ in range(p.num_scales) ]
      result = model.BuildC1(s1)
    self.assertTrue(whiten_replacement.called)
    self.assertEqual(len(result), p.num_scales)
    self.assertTrue(all(x.ndim == 3 for x in result))


class S2Test(unittest.TestCase):

  def testSuccess(self):
    p = Params(num_scales = 2)
    model = Model(p)
    num_kernels = 3
    model._s2_kernels = [ np.random.random((num_kernels,) + shape).astype(
        ACTIVATION_DTYPE) for shape in p.s2_kernel_shapes ]
    c1 = [ np.random.random(p.s2_kernel_shapes[0][:-2] + (100,100)).astype(
        ACTIVATION_DTYPE) for _ in range(p.num_scales) ]
    result = model.BuildS2(c1)
    self.assertEqual(len(result), p.num_scales)
    self.assertTrue(all(len(x) == len(p.s2_kernel_widths) for x in result))
    self.assertTrue(all(all(x.ndim == 3 for x in xs) for xs in result))

  def testAnnotatesError(self):
    p = Params(s2_operation = SLayerOp.RBF, num_scales = 2)
    scale_for_error = 1
    model = Model(BrokenBackend(delay = scale_for_error), p)
    num_kernels = 3
    model._s2_kernels = [ np.random.random((num_kernels,) + shape).astype(
        ACTIVATION_DTYPE) for shape in p.s2_kernel_shapes ]
    c1 = [ np.random.random(p.s2_kernel_shapes[0][:-2] + (100,100)).astype(
        ACTIVATION_DTYPE) for _ in range(p.num_scales) ]
    ex = None
    try:
      model.BuildS2(c1)
    except BackendError, e:
      ex = e
    self.assertIsNotNone(ex)
    self.assertEqual(ex.layer, Layer.S2)
    self.assertIsNotNone(ex.scale, scale_for_error)

  def testErrorWithoutProtos(self):
    p = Params(num_scales = 2)
    model = Model(p)
    c1 = [ np.random.random(p.s2_kernel_shapes[:-2] + [100,100])
        for _ in range(p.num_scales) ]
    with self.assertRaises(Exception):
      model.BuildS2(c1)


class C2Test(unittest.TestCase):

  def testSuccess(self):
    p = Params(num_scales = 2)
    model = Model(p)
    num_protos = 5
    s2 = [ [ np.random.random((num_protos,100,100))
        for _ in range(len(p.s2_kernel_widths)) ] for _ in range(p.num_scales) ]
    result = model.BuildC2(s2)
    self.assertTrue(all(x.ndim == 1 for x in result))
    #~ self.assertEqual(result.ndim, 1)
    self.assertEqual(len(result), len(p.s2_kernel_widths))
    self.assertTrue(all(len(x) == num_protos for x in result))

  def testAnnotatesError(self):
    p = Params(num_scales = 2)
    scale_for_error = 1
    model = Model(BrokenBackend(delay = scale_for_error), p)
    num_protos = 5
    s2 = [ np.random.random((num_protos,1,100,100)).astype(ACTIVATION_DTYPE)
        for _ in range(p.num_scales) ]
    ex = None
    try:
      model.BuildC2(s2)
    except BackendError, e:
      ex = e
    self.assertIsNotNone(ex)
    self.assertEqual(ex.layer, Layer.C2)
    self.assertIsNotNone(ex.scale, scale_for_error)


class EndToEndTest(unittest.TestCase):

  def testBuild_C1FromImage(self):
    model = Model(Params(retina_enabled=True))
    img = np.random.random((100,100)).astype(ACTIVATION_DTYPE)
    state = BuildLayer(model, Layer.C1, model.MakeState(toimage(img)))
    for layer in (Layer.IMAGE, Layer.RETINA, Layer.S1, Layer.C1):
      self.assertIn(layer, state)
      self.assertIsNotNone(state[layer])

  def testBuild_C2FromImage(self):
    p = Params(retina_enabled=True, s2_operation=SLayerOp.RBF)
    model = Model(p)
    num_kernels = 3
    model.s2_kernels = [ np.random.random((num_kernels,) + shape).astype(
        ACTIVATION_DTYPE) for shape in p.s2_kernel_shapes ]
    img = np.random.random((300,300)).astype(ACTIVATION_DTYPE)
    state = BuildLayer(model, Layer.C2, model.MakeState(img))
    for layer in (Layer.IMAGE, Layer.RETINA, Layer.S1, Layer.C1, Layer.S2,
        Layer.C2):
      self.assertIn(layer, state)
      self.assertIsNotNone(state[layer])

  def testBuild_C2FromC1(self):
    p = Params(num_scales=2)
    model = Model(p)
    num_kernels = 3
    model.s2_kernels = [ np.random.random((num_kernels,) + shape).astype(
        ACTIVATION_DTYPE) for shape in p.s2_kernel_shapes ]
    c1 = [ np.random.random(p.s2_kernel_shapes[0][:-2] + (100,100)).astype(
        ACTIVATION_DTYPE) for _ in range(p.num_scales) ]
    state = State({Layer.C1 : c1})
    out_state = BuildLayer(model, Layer.C2, state)
    for layer in (Layer.C1, Layer.S2, Layer.C2):
      self.assertIn(layer, out_state)
      self.assertIsNotNone(out_state[layer])

if __name__ == '__main__':
  unittest.main()
