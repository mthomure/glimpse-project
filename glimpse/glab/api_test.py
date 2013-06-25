import cPickle as pickle
import numpy as np

from . import api
from .api import *
from glimpse.util.gtest import *
from glimpse.experiment import ExpError, ExperimentData, Params
from glimpse import experiment
from glimpse.backends import ACTIVATION_DTYPE

class ApiTests(unittest.TestCase):

  def setUp(self):
    api._vars_obj = None

  def testSetS2Prototypes_withString(self):
    model = GetModel()
    prototypes = [np.random.random((10,) + kshape).astype(ACTIVATION_DTYPE)
        for kshape in model.params.s2_kernel_shapes]
    def load(fh):
      return prototypes
    exp = GetExperiment()
    exp.corpus.paths = []  # avoid "missing corpus" error
    with TempFile() as path:
      with open(path, 'wb') as fh:
        pickle.dump(prototypes, fh)
      SetS2Prototypes(path)
    self.assertEqual(len(model.s2_prototypes), len(prototypes))
    self.assertTrue(all(np.allclose(a,b)
        for a,b in zip(model.s2_prototypes, prototypes)))

  def testSetS2Prototypes_withArrayList(self):
    model = GetModel()
    prototypes = [np.random.random((10,) + kshape).astype(ACTIVATION_DTYPE)
        for kshape in model.params.s2_kernel_shapes]
    exp = GetExperiment()
    exp.corpus.paths = []  # avoid "missing corpus" error
    SetS2Prototypes(prototypes)
    self.assertEqual(len(model.s2_prototypes), len(prototypes))
    self.assertTrue(all(np.allclose(a,b)
        for a,b in zip(model.s2_prototypes, prototypes)))

  def testSetS2Prototypes_withArray(self):
    model = GetModel()
    kshapes = model.params.s2_kernel_shapes
    self.assertEqual(len(kshapes), 1)  # check assumptions
    prototypes = np.random.random((10,) + kshapes[0]).astype(ACTIVATION_DTYPE)
    exp = GetExperiment()
    exp.corpus.paths = []  # avoid "missing corpus" error
    SetS2Prototypes(prototypes)
    self.assertEqual(len(model.s2_prototypes), 1)
    self.assertTrue(np.allclose(model.s2_prototypes[0], prototypes))

  def testSetCorpus_errorIfCorpusExists(self):
    exp = GetExperiment()
    exp.corpus.paths = [ "s%d.png" % i for i in range(4) ]
    with MonkeyPatch(experiment, 'SetCorpus', lambda *args: None):
      with self.assertRaises(ExpError):
        SetCorpus('/path')

  def testImprintS2Prototypes_errorIfNoCorpus(self):
    with MonkeyPatch(api, '_MakePrototypes', lambda *args: None):
      with self.assertRaises(ExpError):
        ImprintS2Prototypes(2)

  def testComputeActivation_errorIfNoCorpus(self):
    with MonkeyPatch(experiment, 'ComputeActivation', lambda *args: None):
      with self.assertRaises(ExpError):
        ComputeActivation()

  def testSetLayer_errorIfNone(self):
    with self.assertRaises(ValueError):
      SetLayer(None)

  def testSetLayer_errorIfActivationExists(self):
    exp = GetExperiment()
    exp.extractor.activation = list()
    with self.assertRaises(ExpError):
      SetLayer("C1")

  def testSetParams_fromKeywords(self):
    SetParams(image_resize_length=13)

  def testSetParams_fromParamsObject(self):
    SetParams(Params(image_resize_length=13))

  def testSetParams_errorIfAlreadySet(self):
    SetParams(Params(image_resize_length=13))
    with self.assertRaises(ExpError):
      SetParams(Params(image_resize_length=14))

  def testLoadParams_errorIfAlreadySet(self):
    SetParams(Params(image_resize_length=13))
    with TempFile() as fname:
      with open(fname, 'wb') as fh:
        pickle.dump(Params(image_resize_length=14), fh)
      with self.assertRaises(ExpError):
        LoadParams(fname)

  def testEvaluateClassifier_errorIfNoCorpus(self):
    with MonkeyPatch(experiment, 'TrainAndTestClassifier',
        lambda *xs, **kw: None):
      with self.assertRaises(ExpError):
        EvaluateClassifier()

  def test_endToEnd(self):
    NUM_PROTOS = 10
    corpus = dict(cls_a = ("11.jpg", "12.jpg"),
        cls_b = ("21.jpg", "22.jpg"))
    with TempDir() as root:
      MakeCorpusWithImages(root, **corpus)
      SetCorpus(root)
      ImprintS2Prototypes(NUM_PROTOS)
      results = EvaluateClassifier()
    self.assertIsNotNone(results.score)

  def test_Reset(self):
    GetExperiment()   # fill _vars_obj
    obj = api._vars_obj
    Reset()
    self.assertNotEqual(obj, api._vars_obj)

  def test_LoadExperiment(self):
    exp = ExperimentData()
    exp.corpus.paths = [ "s%d.png" % i for i in range(4) ]
    with TempFile() as fname:
      with open(fname, 'wb') as fh:
        pickle.dump(exp, fh)
      LoadExperiment(fname)
    self.assertEqual(exp, GetExperiment())

  def test_StoreExperiment(self):
    exp = GetExperiment()
    exp.corpus.paths = [ "s%d.png" % i for i in range(4) ]
    with TempFile() as fname:
      StoreExperiment(fname)
      with open(fname) as fh:
        exp2 = pickle.load(fh)
    self.assertEqual(exp, exp2)
