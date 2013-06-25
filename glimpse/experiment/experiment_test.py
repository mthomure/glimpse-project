from . import experiment
from .experiment import *
from glimpse.models.ml import Model, Params, Layer
from glimpse.pools import SinglecorePool
from glimpse.util.garray import toimage
from glimpse.util.gtest import *

class MiscFuncTests(unittest.TestCase):

  def testResolveLayers(self):
    exp = ExperimentData()
    exp.extractor.model = Model()
    ResolveLayers(exp, 'C2') == [Layer.C2]
    ResolveLayers(exp, Layer.C2) == [Layer.C2]
    ResolveLayers(exp, ('C1', 'C2')) == [Layer.C1, Layer.C2]
    ResolveLayers(exp, ('C1', Layer.C2)) == [Layer.C1, Layer.C2]
    ResolveLayers(exp, (Layer.C1, Layer.C2)) == [Layer.C1, Layer.C2]

  def testVerbose(self):
    from logging import INFO, ERROR
    self.assertEqual(Verbose("true"), INFO)
    self.assertEqual(Verbose("1"), INFO)
    self.assertEqual(Verbose("false"), ERROR)
    self.assertEqual(Verbose("0"), ERROR)
    self.assertEqual(Verbose(True), INFO)
    self.assertEqual(Verbose(False), ERROR)
    import os
    os.environ['GLIMPSE_VERBOSE'] = "1"
    self.assertEqual(Verbose(), INFO)
    os.environ['GLIMPSE_VERBOSE'] = "0"
    self.assertEqual(Verbose(), ERROR)
    del os.environ['GLIMPSE_VERBOSE']
    self.assertEqual(Verbose(), ERROR)

  def testSetModel(self):
    # default
    exp = ExperimentData()
    SetModel(exp)
    self.assertIsNotNone(exp.extractor.model)
    # from model
    model = Model()
    exp = ExperimentData()
    SetModel(exp, model=model)
    self.assertEqual(exp.extractor.model, model)
    # from params
    params = Params()
    exp = ExperimentData()
    SetModel(exp, params=params)
    self.assertEqual(exp.extractor.model.params, params)
    # from keyword params
    exp = ExperimentData()
    SetModel(exp, image_resize_length=10)
    self.assertEqual(exp.extractor.model.params.image_resize_length, 10)


class CorpusFuncTests(unittest.TestCase):

  def testSetCorpus(self):
    corpus = dict(cls_a = tuple(), cls_b = tuple())
    set_corpus_subdirs = RecordedFunctionCall()
    exp = ExperimentData()
    with TempDir() as root:
      MakeCorpus(root, **corpus)
      with MonkeyPatch(experiment, 'SetCorpusSubdirs', set_corpus_subdirs):
        SetCorpus(exp, root)
    self.assertTrue(set_corpus_subdirs.called)
    subdirs = map(os.path.basename, set_corpus_subdirs.args[1])
    self.assertSequenceEqual(subdirs, ('cls_a', 'cls_b'))

  def testSetCorpus_failsOnEmptyDir(self):
    with TempDir() as root:  # test empty corpus root directory
      with self.assertRaises(Exception):
        SetCorpus(exp, root)

  def testSetCorpusSubdirs(self):
    corpus = dict(cls_a = tuple(), cls_b = tuple())
    read_corpus_dirs = RecordedFunctionCall(("paths", "labels"))
    exp = ExperimentData()
    with TempDir() as root:
      MakeCorpus(root, **corpus)
      subdirs = [ os.path.join(root, x) for x in corpus ]
      with MonkeyPatch(experiment, 'ReadCorpusDirs', read_corpus_dirs):
        SetCorpusSubdirs(exp, subdirs)
    self.assertTrue(read_corpus_dirs.called)
    self.assertEqual(exp.corpus.paths, "paths")
    self.assertEqual(exp.corpus.labels, "labels")

  def testSetCorpusSubdirs_withBalance(self):
    corpus = dict(cls_a = tuple(), cls_b = tuple())
    balance_corpus = RecordedFunctionCall(np.ones((2,), np.bool))
    paths = ("img1", "img2", "img3")
    labels = np.array((1,2,2))
    read_corpus_dirs = RecordedFunctionCall((paths, labels))
    exp = ExperimentData()
    with TempDir() as root:
      MakeCorpus(root, **corpus)
      subdirs = [ os.path.join(root, x) for x in corpus ]
      with MonkeyPatch((experiment, 'BalanceCorpus', balance_corpus),
          (experiment, 'ReadCorpusDirs', read_corpus_dirs)):
        SetCorpusSubdirs(exp, subdirs, balance=True)
    self.assertTrue(read_corpus_dirs.called)
    self.assertTrue(balance_corpus.called)
    self.assertSequenceEqual(exp.corpus.paths, ("img1", "img2"))
    self.assertTrue(all(exp.corpus.labels == (1,2)))

  def testSetCorpusSplit_errorOnTrainTestClassDiff(self):
    train_corpus = dict(cls_a = ("a1.jpg", "a2.jpg"),
        cls_b = ("b1.jpg", "b2.jpg"))
    test_corpus = dict(cls_a = ("a3.jpg", "a4.jpg"),
        cls_c = ("c1.jpg", "c2.jpg"))
    with TempDir() as root:
      train_dir = os.path.join(root, 'tr_dir')
      test_dir = os.path.join(root, 'te_dir')
      os.mkdir(train_dir)
      os.mkdir(test_dir)
      MakeCorpus(train_dir, **train_corpus)
      MakeCorpus(test_dir, **test_corpus)
      exp = ExperimentData()
      with self.assertRaises(ExpError):
        SetCorpusSplit(exp, train_dir, test_dir)

  def testSetCorpusSplit(self):
    train_corpus = dict(cls_a = ("a1.jpg", "a2.jpg"),
        cls_b = ("b1.jpg", "b2.jpg"))
    test_corpus = dict(cls_a = ("a3.jpg", "a4.jpg"),
        cls_b = ("b3.jpg", "b4.jpg"))
    with TempDir() as root:
      train_dir = os.path.join(root, 'tr_dir')
      test_dir = os.path.join(root, 'te_dir')
      os.mkdir(train_dir)
      os.mkdir(test_dir)
      MakeCorpus(train_dir, **train_corpus)
      MakeCorpus(test_dir, **test_corpus)
      exp = ExperimentData()
      SetCorpusSplit(exp, train_dir, test_dir)
      i = 0
      for c in (train_corpus['cls_a'], train_corpus['cls_b'],
          test_corpus['cls_a'], test_corpus['cls_b']):
        j = i
        i += len(c)
        self.assertSequenceEqual(sorted(map(os.path.basename,
            exp.corpus.paths[j:i])), c)
      self.assertTrue((exp.corpus.labels == [0,0,1,1]*2).all())
      self.assertTrue((exp.corpus.training_set == [True]*4 + [False]*4).all())


class MakePrototypesTest(unittest.TestCase):

  def test_MakeTrainingExp_corpusTrainingSet(self):
    # should return only training information from original exp
    exp = ExperimentData()
    exp.corpus.paths = ["i%d.png" % i for i in range(8)]
    exp.corpus.labels = np.array([True,False]*4, dtype=bool)
    exp.corpus.training_set = np.array([False]*4 + [True]*4, dtype=bool)
    training_exp = experiment._MakeTrainingExp(exp)
    self.assertSequenceEqual(training_exp.corpus.paths, exp.corpus.paths[4:])
    self.assertTrue((training_exp.corpus.labels == exp.corpus.labels[4:]).all())
    self.assertTrue((training_exp.corpus.training_set == [True]*4).all())
    self.assertIsNone(exp.extractor.training_set)

  def test_MakeTrainingExp_noCorpusTrainingSet(self):
    # should return only training information from original exp
    exp = ExperimentData()
    exp.corpus.paths = np.array(["i%d.png" % i for i in range(8)])
    exp.corpus.labels = np.array([True,False]*4, dtype=bool)
    training_exp = experiment._MakeTrainingExp(exp)
    self.assertIsNotNone(exp.extractor.training_set)
    self.assertTrue((training_exp.corpus.paths ==
        exp.corpus.paths[exp.extractor.training_set]).all())
    self.assertTrue((training_exp.corpus.labels ==
        exp.corpus.labels[exp.extractor.training_set]).all())
    self.assertTrue((training_exp.corpus.training_set == [True]*4).all())

  def test_noTrainingExp(self):
    # no call to make_training_exp(), so corpus information is never pulled
    num_prototypes = 11
    exp = ExperimentData()
    m = exp.extractor.model = Model()
    ks = [ np.random.random((num_prototypes,) + kshape).astype(ACTIVATION_DTYPE)
        for kshape in m.params.s2_kernel_shapes ]
    algorithm = lambda *args,**kw: ks
    MakePrototypes(exp, num_prototypes, algorithm, pool=SinglecorePool())
    for k1,k2 in zip(ks, exp.extractor.model.s2_kernels):
      self.assertTrue((k1 == k2).all())
    self.assertIsNone(exp.extractor.training_set)
    self.assertEquals(algorithm, exp.extractor.prototype_algorithm)

  def test_algorithmLocations(self):
    num_prototypes = 11
    exp = ExperimentData()
    m = exp.extractor.model = Model()
    ks = [ np.random.random((11,) + kshape)
        for kshape in m.params.s2_kernel_shapes ]
    algorithm = lambda *xs: ks
    # Use image indices 1 and 3 for algorithm, relative to training set. This is
    # indices 5 and 7, relative to full image set.
    algorithm.locations = [np.array([[1,0,0,0],[3,0,0,0]])] \
        * len(m.params.s2_kernel_shapes)
    # no call to make_training_exp(), so need to set it manually
    exp.corpus.training_set = np.array([False]*4 + [True]*4, dtype=bool)
    MakePrototypes(exp, num_prototypes, algorithm, pool=SinglecorePool())
    self.assertEqual(len(algorithm.locations), len(ks))
    for locs in algorithm.locations:
      # algorithm used image indices 4-7 (relative to full image set)
      self.assertTrue((locs[:,0] == (5,7)).all())


class ComputeActivationTests(unittest.TestCase):

  def testComputeActivationMaps_singleLayer(self):
    m = Model()
    img = toimage(np.random.random((100,100)) * 255)
    exp = ExperimentData()
    exp.corpus.paths = [img]
    exp.extractor.model = Model()
    ComputeActivation(exp, Layer.C1, SinglecorePool(), save_all=True)
    states = exp.extractor.activation
    self.assertEqual(len(states), 1)
    self.assertEqual(len(states[0]), 4)
    for layer in (Layer.IMAGE, Layer.RETINA, Layer.S1, Layer.C1):
      self.assertIn(layer, states[0])

  def testComputeActivationMaps_multipleLayers(self):
    m = Model()
    img1 = toimage(np.random.random((100,100)) * 255)
    img2 = toimage(np.random.random((100,100)) * 255)
    exp = ExperimentData()
    exp.corpus.paths = [img1, img2]
    exp.extractor.model = Model()
    ComputeActivation(exp, Layer.C1, SinglecorePool(), save_all=True)
    states = exp.extractor.activation
    self.assertEqual(len(states), 2)
    for state in states:
      self.assertEqual(len(state), 4)
      for layer in (Layer.IMAGE, Layer.RETINA, Layer.S1, Layer.C1):
        self.assertIn(layer, state)

  def testComputeActivationMaps_noSaveAll(self):
    m = Model()
    img = toimage(np.random.random((100,100)) * 255)
    exp = ExperimentData()
    exp.corpus.paths = [img]
    exp.extractor.model = Model()
    ComputeActivation(exp, Layer.C1, SinglecorePool(), save_all=False)
    states = exp.extractor.activation
    self.assertEqual(len(states), 1)
    self.assertEqual(len(states[0]), 1)
    self.assertIn(Layer.C1, states[0])


class TrainAndTestClassifierTests(unittest.TestCase):

  # TODO: add tests for
  #  - calls score_func and feature_builder callbacks
  #  - respects learner and train_size parameters

  NUM_INSTANCES = 100

  def _buildHelper(self, exp, layers, **kw):
    exp.extractor.model = Model()
    # Provide 12 features per instance
    extractor = lambda *args, **kw: np.random.random((self.NUM_INSTANCES, 12))
    exp.corpus.paths = [ "img%d.jpg" % x for x in range(self.NUM_INSTANCES) ]
    exp.corpus.labels = np.array([1]*(self.NUM_INSTANCES/2) +
        [2]*(self.NUM_INSTANCES/2))
    exp.extractor.activation = list()  # fake activation list
    with MonkeyPatch(experiment, 'ExtractFeatures', extractor):
      TrainAndTestClassifier(exp, layers, **kw)
    self.assertEqual(len(exp.evaluation), 1)
    eval_ = exp.evaluation[0]
    self.assertIn('classifier', eval_.results)
    self.assertIn('score', eval_.results)
    self.assertIn('score_func', eval_.results)
    self.assertIn('training_score', eval_.results)

  def testFixedSplit_newTrainSet(self):
    exp = ExperimentData()
    self._buildHelper(exp, layers="c2")
    eval_ = exp.evaluation[0]
    self.assertSequenceEqual(eval_.training_set.shape, (self.NUM_INSTANCES,))

  def testFixedSplit_trainSetInCorpus(self):
    exp = ExperimentData()
    exp.corpus.training_set = np.array([True] +
        [False]*(self.NUM_INSTANCES/2-1) + [True] +
        [False]*(self.NUM_INSTANCES/2-1), dtype = np.bool)
    self._buildHelper(exp, layers="c2")
    eval_ = exp.evaluation[0]
    self.assertIsNone(exp.extractor.training_set)

  def testFixedSplit_trainSetInExtractor(self):
    exp = ExperimentData()
    exp.extractor.training_set = np.array([True] +
        [False]*(self.NUM_INSTANCES/2-1) + [True] +
        [False]*(self.NUM_INSTANCES/2-1), dtype = np.bool)
    self._buildHelper(exp, layers ="c2")
    eval_ = exp.evaluation[0]
    self.assertIsNone(eval_.training_set)


class CrossValidateClassifierTests(unittest.TestCase):

  # TODO: add tests for
  #  - respects learner, feature_builder, score_func
  #  - fails on #images < #folds

  NUM_INSTANCES = 100

  def _buildHelper(self, exp, layers, **kw):
    exp.extractor.model = Model()
    # Provide 12 features per instance
    extractor = lambda *args, **kw: np.random.random((self.NUM_INSTANCES, 12))
    exp.corpus.paths = [ "img%d.jpg" % x for x in range(self.NUM_INSTANCES) ]
    exp.corpus.labels = np.array([1]*(self.NUM_INSTANCES/2) +
        [2]*(self.NUM_INSTANCES/2))
    with MonkeyPatch(experiment, 'ExtractFeatures', extractor):
      CrossValidateClassifier(exp, layers, **kw)
    self.assertEqual(len(exp.evaluation), 1)

  def test_errorOnCorpusTrainingSet(self):
    exp = ExperimentData()
    # fails because cross-val can't use training set
    exp.corpus.training_set = np.array((True, False, True, False),
        dtype = np.bool)
    with self.assertRaises(ExpError):
      self._buildHelper(exp, layers="c2")

  def test_errorOnExtractorTrainingSet(self):
    exp = ExperimentData()
    # fails because cross-val can't use training set
    exp.extractor.training_set = np.array((True, False, True, False),
        dtype = np.bool)
    with self.assertRaises(ExpError):
      self._buildHelper(exp, layers="c2")

  def testCrossval(self):
    exp = ExperimentData()
    self._buildHelper(exp, layers="c2")
    eval_ = exp.evaluation[0]
    self.assertTrue(eval_.results.cross_validate)
    self.assertIsNotNone(eval_.results.score)
    self.assertIsNone(eval_.training_set)

class EndToEndTests(unittest.TestCase):

  def testEndToEnd(self):
    NUM_PROTOS = 10
    exp = ExperimentData()
    exp.extractor.model = Model()
    with TempDir() as root:
      MakeCorpusWithImages(root, a=('a1.png', 'a2.png'), b=('b1.png', 'b2.png'))
      SetCorpus(exp, root)
      MakePrototypes(exp, NUM_PROTOS, 'imprint')
      ComputeActivation(exp, Layer.C2, SinglecorePool())
    TrainAndTestClassifier(exp, Layer.C2)
    self.assertEqual(len(exp.corpus.paths), 4)
    self.assertTrue((exp.corpus.labels == (0, 0, 1, 1)).all())
    self.assertEqual(exp.extractor.training_set.sum(), 2)
    self.assertEqual(len(exp.extractor.model.s2_kernels[0]), NUM_PROTOS)
    st = exp.extractor.activation[0]
    self.assertEqual(len(exp.extractor.activation), 4)
    self.assertIn(Layer.C2, st)
    self.assertEqual(len(st[Layer.C2][0]), NUM_PROTOS)
    self.assertEqual(len(exp.evaluation), 1)
    self.assertIsNotNone(exp.evaluation[0].results.score)

if __name__ == '__main__':
  unittest.main()
