from . import cli
from .cli import *
from glimpse.models.ml import Model, Params
from glimpse.pools import SinglecorePool
from glimpse.util.garray import toimage
from glimpse.util.gtest import *

# helper function
def MakeOpts():
  return OptValue(MakeCliOptions())

class CliWithActivityTests(unittest.TestCase):

  def testLoadParams(self):
    # Set some non-default parameters
    p = Params(num_scales = 52, s1_num_orientations = 13)
    pool = SinglecorePool()
    exp = ExperimentData()
    opts = MakeOpts()
    opts.extractor.no_activity = True
    with TempFile() as path:
      with open(path, 'w') as fh:
        pickle.dump(p, fh)
      opts.extractor.param_path = path
      CliWithActivity(opts, exp, pool)
    self.assertEqual(exp.extractor.model.params, p)

  def testLoadPrototypes(self):
    m = Model()
    ks = [ np.random.random((10,) + kshape).astype(ACTIVATION_DTYPE)
        for kshape in m.params.s2_kernel_shapes ]
    exp = ExperimentData()
    exp.extractor.model = m
    opts = MakeOpts()
    opts.extractor.no_activity = True
    with TempFile() as path:
      with open(path, 'w') as fh:
        pickle.dump(ks, fh)
      opts.extractor.prototypes.path = path
      CliWithActivity(opts, exp, SinglecorePool())
    self.assertEqual(len(exp.extractor.model.s2_kernels), len(ks))
    for s2_k, k in zip(exp.extractor.model.s2_kernels, ks):
      self.assertTrue(np.all(s2_k == k))

  def test_withActivity(self):
    opts = MakeOpts()
    f = RecordedFunctionCall()
    with MonkeyPatch(cli, 'ComputeActivation', f):
      CliWithActivity(opts, ExperimentData(), SinglecorePool())
    self.assertTrue(f.called)

class CliEvaluateTests(unittest.TestCase):

  def test_errorOnMissingLayer(self):
    opts = MakeOpts()
    opts.evaluation.layer = None
    with self.assertRaises(OptionError):
      CliEvaluate(opts, ExperimentData())

class CliProjectTests(unittest.TestCase):

  def test_loadInputExp(self):
    iexp = ExperimentData()
    iexp.corpus.paths = [ "a%d.png" % i for i in range(4) ]
    opts = MakeOpts()
    opts.extractor.no_activity = True
    with TempFile() as ifname:
      with open(ifname, 'wb') as fh:
        pickle.dump(iexp, fh)
      opts.input_path = ifname
      exp = CliProject(opts)
    self.assertEqual(exp, iexp)

  def test_errorOnBadCorpus(self):
    opts = MakeOpts()
    opts.corpus.root_dir = '/does/not/exist'
    with self.assertRaises(OSError):
      CliProject(opts)

  def testEndToEnd_evaluate(self):
    corpus = dict(cls_a = ("11.jpg", "12.jpg"),
        cls_b = ("21.jpg", "22.jpg"))
    opts = MakeOpts()
    opts.evaluation.evaluate = True
    opts.evaluation.layer = "c1"
    with TempDir() as root:
      MakeCorpusWithImages(root, **corpus)
      opts.corpus.root_dir = root
      exp = CliProject(opts)
    self.assertIsNotNone(exp)
    self.assertEqual(len(exp.evaluation), 1)
    self.assertIsNotNone(exp.evaluation[0].results.score)

  def testEndToEnd_noEvaluate(self):
    corpus = dict(cls_a = ("11.jpg", "12.jpg"),
        cls_b = ("21.jpg", "22.jpg"))
    opts = MakeOpts()
    opts.evaluation.evaluate = False
    opts.evaluation.layer = "c1"
    with TempDir() as root:
      MakeCorpusWithImages(root, **corpus)
      opts.corpus.root_dir = root
      exp = CliProject(opts)
    self.assertIsNotNone(exp)
    self.assertEqual(len(exp.evaluation), 0)

  def testSetPool(self):
    f = RecordedFunctionCall()
    opts = MakeOpts()
    opts.pool_type = 'MY-CRAZY-POOL'
    opts.corpus.root_dir = '1'
    with MonkeyPatch((cli, 'MakePool', f),
        (cli, 'CliWithActivity', lambda *x: None),
        (cli, 'SetCorpus', lambda *x,**kw: None)):
      CliProject(opts)
    self.assertSequenceEqual(f.args, (opts.pool_type,))

  def testWriteResults(self):
    opts = MakeOpts()
    opts.extractor.no_activity = True
    opts.corpus.root_dir = '1'
    with TempFile() as fname:
      opts.result_path = fname
      with MonkeyPatch(cli, 'SetCorpus', lambda *args, **kw: None):
        CliProject(opts)
      with open(fname) as fh:
        exp = pickle.load(fh)
    self.assertEqual(type(exp), ExperimentData)

import subprocess
import cPickle as pickle

class ShellTests(unittest.TestCase):

  def test(self):
    corpus = dict(cls_a=('a1.png', 'a2.png'), cls_b=('b1.png', 'b2.png'))
    with TempDir() as root:
      MakeCorpusWithImages(root, **corpus)
      with TempFile() as fname:
        cmd = ["python", "-m", "glimpse.glab.cli", "-p", "imprint",
            "-E", "-c", root, "-o", fname, "-E", "-t", "singlecore"]
        subprocess.check_call(cmd)
        with open(fname) as fh:
          exp = pickle.load(fh)
    self.assertEqual(len(exp.corpus.paths), 4)
    self.assertTrue((exp.corpus.labels == (0,0,1,1)).all())
    self.assertEqual(len(exp.extractor.activation), 4)
    self.assertEqual(len(exp.evaluation), 1)
    self.assertIsNotNone(exp.evaluation[0].results.score)

if __name__ == '__main__':
  unittest.main()
