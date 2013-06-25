from glimpse.util.gtest import *
from glimpse.backends import ACTIVATION_DTYPE

from . import prototype_algorithms
from .prototype_algorithms import *
from .prototype_algorithms import _SimpleLearningAlg
from .experiment import ExperimentData
from glimpse.models.ml import Model, Params

# TODO: add tests for weighted and unweighted k-Means

class AlgTests(unittest.TestCase):

  def testImprintAlg_default(self):
    def make_training_exp():
      exp = ExperimentData()
      exp.corpus.paths = list()
      return exp
    model = Model()
    f = RecordedFunctionCall(('PATCHES', None))
    with MonkeyPatch(prototype_algorithms, 'SamplePatchesFromImages', f):
      alg = ImprintAlg()
      patches_per_shape_ = alg(11,  # anything numeric
          model, make_training_exp, None, None)
    self.assertEqual(patches_per_shape_, 'PATCHES')

  def testImprintAlg_recordLocations(self):
    locs_per_shape = 'LOCS'  # some value to test for
    def make_training_exp():
      exp = ExperimentData()
      exp.corpus.paths = list()
      return exp
    patches_per_shape = ['PATCHES%d' % i for i in range(2)]  # two kernel widths
    model = Model(Params(s2_kernel_widths=(7,9)))
    f = RecordedFunctionCall((patches_per_shape, locs_per_shape))
    with MonkeyPatch(prototype_algorithms, 'SamplePatchesFromImages', f):
      alg = ImprintAlg(True)
      patches_per_shape_ = alg(11,  # anything numeric
          model, make_training_exp, None, None)
    self.assertEqual(len(patches_per_shape_), 2)
    self.assertSequenceEqual(patches_per_shape, patches_per_shape_)
    #~ self.assertTrue(all((ps1 == ps2).all() for ps1,ps2 in zip(patches_per_shape,
        #~ patches_per_shape_)))
    self.assertEqual(locs_per_shape, alg.locations)

  def testImprintAlg_withNorm(self):
    locs_per_shape = 'LOCS'  # some value to test for
    def make_training_exp():
      exp = ExperimentData()
      exp.corpus.paths = list()
      return exp
    model = Model(Params(s2_kernel_widths=(7,9), s2_operation='NormRbf'))
    # These patches don't match C1 structure, but that's fine. We just want to
    # test that normalizing multi-dimensional arrays works when imprinting.
    patches_per_shape = [ np.random.random((3,4,w,w)) for w in (7,9) ]
    f = RecordedFunctionCall((patches_per_shape, locs_per_shape))
    with MonkeyPatch(prototype_algorithms, 'SamplePatchesFromImages', f):
      alg = ImprintAlg()
      patches_per_shape_ = alg(11,  # anything numeric
          model, make_training_exp, None, None)
    self.assertEqual(len(patches_per_shape_), 2)
    self.assertTrue(all((ps1 == ps2).all() for ps1,ps2 in zip(patches_per_shape,
        patches_per_shape_)))
    self.assertEqual(locs_per_shape, alg.locations)

  def testShuffledAlg_default(self):
    def make_training_exp():
      exp = ExperimentData()
      exp.corpus.paths = list()
      return exp
    model = Model()
    # These patches don't match C1 structure, but that's fine. We just want to
    # test that shuffling the imprinted array data works.
    patches_per_shape = [ np.random.random((3,4,w,w))
        for w in model.params.s2_kernel_widths ]
    f = RecordedFunctionCall((patches_per_shape, None))
    with MonkeyPatch(prototype_algorithms, 'SamplePatchesFromImages', f):
      alg = ShuffledAlg()
      patches_per_shape_ = alg(11,  # anything numeric
          model, make_training_exp, None, None)
    self.assertEqual(len(patches_per_shape_),
        len(model.params.s2_kernel_widths))
    for ps1,ps2 in zip(patches_per_shape, patches_per_shape_):
      for p1,p2 in zip(ps1,ps2):
        self.assertSequenceEqual(sorted(p1.flat), sorted(p2.flat))

  def testShuffledAlg_twoKernelWidths(self):
    def make_training_exp():
      exp = ExperimentData()
      exp.corpus.paths = list()
      return exp
    model = Model(Params(s2_kernel_widths=(7,9), s2_operation='NormRbf'))
    # These patches don't match C1 structure, but that's fine. We just want to
    # test that shuffling the imprinted array data works.
    patches_per_shape = [ np.random.random((3,4,w,w)) for w in (7,9) ]
    f = RecordedFunctionCall((patches_per_shape, None))
    with MonkeyPatch(prototype_algorithms, 'SamplePatchesFromImages', f):
      alg = ShuffledAlg()
      patches_per_shape_ = alg(11,  # anything numeric
          model, make_training_exp, None, None)
    self.assertEqual(len(patches_per_shape_), 2)
    for ps1,ps2 in zip(patches_per_shape, patches_per_shape_):
      for p1,p2 in zip(ps1,ps2):
        self.assertSequenceEqual(sorted(p1.flat), sorted(p2.flat))

  def testUniformAlg_default(self):
    num_prototypes = 11
    model = Model()
    alg = UniformAlg()
    patches_per_shape = alg(num_prototypes, model, None, None, None)
    self.assertEqual(len(patches_per_shape), len(model.params.s2_kernel_widths))
    for ps,kshape in zip(patches_per_shape, model.params.s2_kernel_shapes):
      self.assertEqual(ps.dtype, ACTIVATION_DTYPE)
      self.assertSequenceEqual(ps.shape, (num_prototypes,) + kshape)

  def testUniformAlg_withNorm(self):
    num_prototypes = 11
    model = Model(Params(s2_kernel_widths=(7,9), s2_operation='NormRbf'))
    alg = UniformAlg()
    patches_per_shape = alg(num_prototypes, model, None, None, None)
    self.assertEqual(len(patches_per_shape), len(model.params.s2_kernel_widths))
    for ps,kshape in zip(patches_per_shape, model.params.s2_kernel_shapes):
      self.assertEqual(ps.dtype, ACTIVATION_DTYPE)
      self.assertSequenceEqual(ps.shape, (num_prototypes,) + kshape)

  def testUniformAlg_customLimits(self):
    num_prototypes = 11
    low = -1
    high = 10
    model = Model(Params(s2_kernel_widths=(7,)))
    f = RecordedFunctionCall('PATCHES')
    with MonkeyPatch('glimpse.prototypes', 'UniformRandom', f):
      alg = UniformAlg(low=low, high=high)
      patches_per_shape = alg(num_prototypes, model, None, None, None)
    self.assertEqual(patches_per_shape, ['PATCHES'] *
        len(model.params.s2_kernel_widths))
    self.assertTrue(f.called)
    self.assertSequenceEqual(f.args[0:1] + f.args[2:4], (num_prototypes, low,
        high))

  def test_SimpleLearningAlg_default(self):
    num_prototypes = 11
    patches_per_shape = 'PATCHES'
    learner = 'LEARNER'
    model = Model()
    def make_training_exp():
      exp = ExperimentData()
      exp.corpus.paths = list()
      return exp
    f = RecordedFunctionCall(patches_per_shape)
    with MonkeyPatch('glimpse.prototypes', 'SampleAndLearnPatches', f):
      alg = _SimpleLearningAlg(learner)
      patches_per_shape_ = alg(num_prototypes, model, make_training_exp, None,
          None)
    self.assertSequenceEqual(f.args[2:4], (learner,num_prototypes))

  def test_SimpleLearningAlg_withNorm(self):
    num_prototypes = 3
    # These patches don't match C1 structure, but that's fine. We just want to
    # test normalization of patch data.
    patches_per_shape = [ np.random.random((3,4,w,w)) for w in (7,9) ]
    learner = 'LEARNER'
    model = Model()
    def make_training_exp():
      exp = ExperimentData()
      exp.corpus.paths = list()
      return exp
    f = RecordedFunctionCall(patches_per_shape)
    with MonkeyPatch('glimpse.prototypes', 'SampleAndLearnPatches', f):
      alg = _SimpleLearningAlg(learner)
      patches_per_shape_ = alg(num_prototypes, model, make_training_exp, None,
          None)
    self.assertEqual(len(patches_per_shape_), len(patches_per_shape))
    for ps,kshape in zip(patches_per_shape_, model.params.s2_kernel_shapes):
      self.assertSequenceEqual(ps.shape, (num_prototypes,) + kshape)

  def test_SimpleLearningAlg_withNumSamples(self):
    num_prototypes = 11
    num_samples = 23
    patches_per_shape = 'PATCHES'
    learner = 'LEARNER'
    model = Model()
    def make_training_exp():
      exp = ExperimentData()
      exp.corpus.paths = list()
      return exp
    f = RecordedFunctionCall(patches_per_shape)
    with MonkeyPatch('glimpse.prototypes', 'SampleAndLearnPatches', f):
      alg = _SimpleLearningAlg(learner)
      alg.num_samples = num_samples
      patches_per_shape_ = alg(num_prototypes, model, make_training_exp, None,
          None)
    self.assertSequenceEqual(f.args[2:4], (learner,num_prototypes))
    self.assertEqual(f.kw['num_samples'], num_samples)

  def testGetAlgorithmNames(self):
    names = GetAlgorithmNames()
    for name in ('imprint', 'uniform', 'shuffle', 'histogram', 'normal',
        'kmeans', 'nearest_kmeans', 'kmedoids', 'pca', 'ica', 'nmf',
        'sparse_pca'):
      self.assertIn(name, names)

  def testResolveAlgorithm(self):
    self.assertEqual(ResolveAlgorithm('imprint'), ImprintAlg)
    self.assertEqual(ResolveAlgorithm('histogram'), HistogramAlg)
