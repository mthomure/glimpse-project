"""Provides a simple, declarative-like interface for running experiments."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from copy import copy
import inspect
import logging
import cPickle as pickle

from decorator import decorator
from glimpse.models import MakeModel, MakeParams
from glimpse.pools import MakePool
from glimpse import experiment
from glimpse.experiment import ExpError
from glimpse.util.progress import ProgressBar
from glimpse.util import docstring

from glimpse.experiment import misc

# Constants used by @depends and @provides
CORPUS = 'corpus'
MODEL = 'model'
PROTOTYPES = 'prototypes'
ACTIVATION = 'activation'
EVALUATION = 'evaluation'

#: Default model layer to use for evaluation.
DEFAULT_LAYER = "C2"

class ApiVars(object):
  # once model exists, parameters can not be altered
  # once features or prototypes exist, model can not be altered

  exp = None  # current experiment data
  reader = None  # file and directory name reader
  verbose = False
  _pool = None  # worker pool to use when computing model activity
  _params = None  # parameters to use for model
  _layer = DEFAULT_LAYER  # model layer to use for classifier features

  def __init__(self):
    self.exp = experiment.ExperimentData()
    self.reader = experiment.DirReader()

  @property
  def pool(self):
    if self._pool is None:
      self._pool = MakePool()
      logging.info("Using pool: %s" % type(self._pool).__name__)
    return self._pool

  @property
  def params(self):
    if self._params is None:
      self._params = MakeParams()
    return self._params

  @params.setter
  def params(self, value):
    if value == self._params:
      return  # noop
    if self._params is not None:
      raise ExpError("Parameters already set -- call Reset() to change them")
    self._params = value

  @property
  def layer(self):
    return self._layer

  @layer.setter
  def layer(self, value):
    if value == self._layer:
      return  # noop
    if self.exp.extractor.activation is not None:
      raise ExpError("Can't set layer after activation has been computed")
    self._layer = value

  def Store(self, path):
    with open(path, 'wb') as fh:
      pickle.dump(self.exp, fh, protocol = -1)

  @staticmethod
  def Load(path):
    vs = ApiVars()
    with open(path, 'rb') as fh:
      exp = pickle.load(fh)
    if exp.extractor.model:
      vs._params = exp.extractor.model.params  # mark parameters as fixed
    vs.exp = exp
    return vs

  ### METHODS TO SUPPORT DEPENDS/PROVIDES ###

  def AssertCorpus(self, flag):
    if flag != (self.exp and self.exp.corpus.paths is not None):
      raise ExpError(("Corpus information has been set already",
          "Corpus information is missing -- please call one of the SetCorpus() "
          "functions first")[flag])

  def AssertPrototypes(self, flag):
    if flag != (self.exp and self.exp.extractor.model and
        self.exp.extractor.model.s2_kernels is not None):
      raise ExpError(("Prototypes have been set already",
          "Prototypes are missing")[flag])

  def HasActivation(self):
    return self.exp and self.exp.extractor.activation is not None

  def AssertActivation(self, flag):
    if flag != self.HasActivation():
      raise ExpError(("Model activation has been computed already",
          "Model activation is missing -- please call ComputeActivation() "
          "first")[flag])

  def AssertEvaluation(self, flag):
    if flag != (self.exp and len(self.exp.evaluation) > 0):
      raise ExpError(("Model has been evaluated already",
          "Model has not been evaluated -- please call EvaluateClassifier() "
          "first")[flag])

  def AssertModel(self, flag):
    if flag != (self.exp and self.exp.model is not None):
      raise ExpError(("Model has been set already",
          "Model is missing")[flag])

  def EnsureModel(self):
    if not self.exp.extractor.model:
      self.exp.extractor.model = MakeModel(self._params)
      self._params = self.exp.extractor.model.params  # mark parameters as fixed
    return self.exp.extractor.model

def _DependsDecorator(flag, ids):
  if len(ids) == 1 and inspect.isfunction(ids[0]):
    # If given a function, user must have forgotten the decorator argument.
    raise ValueError("decorator takes an argument")
  def wrap(f, *args, **kw):
    vs = _vars()
    for id in ids:
      if id == CORPUS:
        vs.AssertCorpus(flag)
      elif id == MODEL:
        if flag:
          vs.EnsureModel()
        else:
          vs.AssertModel(False)
      elif id == PROTOTYPES:
        vs.AssertPrototypes(flag)
      elif id == ACTIVATION:
        vs.AssertActivation(flag)
      elif id == EVALUATION:
        vs.AssertEvaluation(flag)
    return f(*args, **kw)
  def dec(f):
    return decorator(wrap, f)
  return dec

def depends(*ids):
  """Decorator that ensures a given values exists.

  Example:

  >>> @depends(CORPUS)
  >>> def my_corpus_getter():
  >>>   return exp.corpus.paths

  """
  return _DependsDecorator(True, ids)

def provides(*ids):
  """Decorator that ensures a given value is *not* set.

  This is the reverse of the @depends decorator.

  Example:

  >>> @provides(CORPUS)
  >>> def my_corpus_setter(paths):
  >>>   exp.corpus.paths = paths

  """
  return _DependsDecorator(False, ids)

_vars_obj = None
def _vars():
  global _vars_obj
  if _vars_obj is None:
    _vars_obj = ApiVars()
  return _vars_obj

def Reset():
  """Clear all experimental settings and results."""
  global _vars_obj
  _vars_obj = None

def SetParams(params=None, **kw):
  """Choose model parameters."""
  if params is None:
    params = MakeParams(**kw)
  else:
    params = copy(params)
    for k,v in kw.items():
      params[k] = v
  _vars().params = params

def SetParamsWithGui(params=None, **kw):
  """Choose model parameters using a graphical interface."""
  if params is None:
    params = MakeParams(**kw)
  else:
    params = copy(params)
    for k,v in kw.items():
      params[k] = v
  if params.configure_traits():
    _vars().params = params

def LoadParams(path):
  """Read model parameters from disk."""
  logging.info("Reading model parameters from file: %s" % path)
  with open(path) as fh:
    _vars().params = pickle.load(fh)

def SetLayer(layer):
  """Set the model layer to use for evaluation."""
  if layer is None:
    raise ValueError
  _vars().layer = layer

def StoreExperiment(path):
  """Store settings and results for experiment to disk."""
  logging.info("Writing experiment data to file -- %s" % path)
  _vars().Store(path)

def LoadExperiment(path):
  """Load settings and results for experiment from disk."""
  global _vars_obj
  logging.info("Reading experiment data from file -- %s" % path)
  _vars_obj = ApiVars.Load(path)

def GetExperiment():
  """Get the current experiment object.

  This is an advanced function. In general, the user should not modify the
  experiment object directly.

  """
  return _vars().exp

@docstring.copy_dedent(experiment.Verbose)
def Verbose(flag=True):
  experiment.Verbose(flag)
  _vars().verbose = flag

def GetModel():
  """Get the Glimpse model used for this experiment.

  This is an advanced function. In general, the user should not need to interact
  with the model directly.

  """
  return _vars().EnsureModel()

### CORPUS ###

@docstring.copy_dedent(experiment.SetCorpus)
@provides(CORPUS)
def SetCorpus(corpus_dir, balance=False):
  vs = _vars()
  experiment.SetCorpus(vs.exp, corpus_dir, balance, vs.reader)

@docstring.copy_dedent(experiment.SetCorpusSubdirs)
@provides(CORPUS)
def SetCorpusSubdirs(corpus_subdirs, balance=False):
  vs = _vars()
  experiment.SetCorpusSubdirs(vs.exp, corpus_subdirs, balance, vs.reader)

@docstring.copy_dedent(experiment.SetCorpusSplit)
@provides(CORPUS)
def SetCorpusSplit(train_dir, test_dir):
  vs = _vars()
  experiment.SetCorpusSplit(vs.exp, train_dir, test_dir, vs.reader)

def SetCorpusByName(name):
  """Use a sample image corpus for this experiment.

  :param str name: Corpus name. One of 'easy', 'moderate', or 'hard'.

  This provides access to a small set of images for demonstration purposes,
  which are composed of simple shapes on various background patterns.

  """
  SetCorpus(experiment.GetCorpusByName(name))

### S2 PROTOTYPES ###

@depends(CORPUS, MODEL)
@provides(PROTOTYPES)
def SetS2Prototypes(prototypes):
  """Manually specify the set of S2 prototypes.

  :type prototypes: str or list of array of float
  :param prototypes: Path to prototypes on disk, or prototypes array to set.
  :rtype: list of array of float
  :return: Set of model prototypes.

  """
  vs = _vars()
  model = vs.exp.extractor.model
  if isinstance(prototypes, basestring):
    path = prototypes
    with open(path) as fh:
      prototypes = pickle.load(fh)
    # XXX assumes one prototype size
    logging.info("Read %d prototypes from file: %s", len(prototypes[0]),
        path)
  else:
    if (not isinstance(prototypes, list) and
        len(model.params.s2_kernel_widths) == 1):
      # assume that prototypes were passed for a single kernel size
      prototypes = [prototypes]
    # XXX assumes one prototype size
    logging.info("Manually setting %d prototypes", len(prototypes[0]))
  model.s2_prototypes = prototypes
  return prototypes

@depends(MODEL)
@provides(PROTOTYPES)
def _MakePrototypes(num_prototypes, alg):
  vs = _vars()
  if vs.verbose:
    progress = ProgressBar
  else:
    progress = None
  experiment.MakePrototypes(vs.exp, num_prototypes, alg, vs.pool,
      progress = progress)

@depends(CORPUS)
def ImprintS2Prototypes(num_prototypes):
  """Create a set of S2 prototypes by "imprinting" from training images.

  Patches are drawn from all classes of the training data.

  :param int num_prototypes: Number of prototypes to create.

  """
  alg = experiment.ImprintProtoAlg()
  _MakePrototypes(num_prototypes, alg)

def MakeUniformRandomS2Prototypes(num_prototypes, low=None, high=None):
  """Create a set of random S2 prototypes drawn from the uniform distribution.

  Each element of every prototype is drawn independently from an uniform
  distribution with the same parameters.

  :param int num_prototypes: Number of prototypes to create.
  :param float low: Minimum value in uniform range.
  :param float high: Maximum value in uniform range.

  """
  alg = experiment.UniformProtoAlg(low, high)
  _MakePrototypes(num_prototypes, alg)

@depends(CORPUS)
def MakeShuffledRandomS2Prototypes(num_prototypes):
  """Create a set of "imprinted" S2 prototypes that have been shuffled.

  Each prototype has its contents randomly permuted across location and
  orientation band.

  :param int num_prototypes: Number of prototypes to create.

  """
  alg = experiment.ShuffledProtoAlg()
  _MakePrototypes(num_prototypes, alg)

@depends(CORPUS)
def MakeHistogramRandomS2Prototypes(num_prototypes):
  """Create a set of S2 prototypes drawn from a 1D histogram of C1 activity.

  The set is created by drawing elements from a distribution that is estimated
  from a set of imprinted prototypes. Each entry is drawn independently of the
  others.

  :param int num_prototypes: Number of prototypes to create.

  """
  alg = experiment.HistogramProtoAlg()
  _MakePrototypes(num_prototypes, alg)

@depends(CORPUS)
def MakeNormalRandomS2Prototypes(num_prototypes):
  """Create a set of random S2 prototypes drawn from the normal distribution.

  Each element of every prototype is drawn independently from a normal
  distribution with the same parameters.

  :param int num_prototypes: Number of prototypes to create.

  """
  alg = experiment.NormalProtoAlg()
  _MakePrototypes(num_prototypes, alg)

@depends(CORPUS)
def MakeKmeansS2Prototypes(num_prototypes, num_patches=None):
  """Create a set of S2 prototypes by clustering C1 samples with k-Means.

  :param int num_prototypes: Number of prototypes to create.
  :param int num_patches: Number of sample patches passed to k-Means.

  """
  alg = experiment.KmeansProtoAlg(num_patches)
  _MakePrototypes(num_prototypes, alg)

### EVALUATION ###

@docstring.copy_dedent(experiment.ComputeActivation)
@depends(CORPUS, MODEL)
@provides(ACTIVATION)
def ComputeActivation(save_all=False):
  vs = _vars()
  if vs.verbose:
    progress = ProgressBar
  else:
    progress = None
  experiment.ComputeActivation(vs.exp, vs.layer, vs.pool, save_all=save_all,
      progress=progress)

@depends(MODEL)
def GetImageFeatures(images):
  """Compute model features for images not in the experiment's corpus.

  :type path: str or list of str
  :param path: Filesystem path for one or more images.
  :rtype: 2D array of float
  :return: Feature vector for image, with rows and columns corresponding to
     images and features, respectively.

  """
  if isinstance(images, basestring):
    images = [images]
  vs = _vars()
  if vs.verbose:
    progress = ProgressBar
  else:
    progress = None
  return experiment.GetImageFeatures(vs.exp, vs.layer, images, vs.pool,
      progress=progress)

def EvaluateClassifier(cross_validate=False, algorithm=None):
  """Apply a classifier to the image features in the experiment.

  :param bool cross_validate: Whether to use cross-validation. The default will
     use a fixed training and testing split.
  :param learner: Learning algorithm, which is fit to features. This should be a
     scikit-learn classifier object. If not set, a linear SVM is used.

  """
  vs = _vars()
  if not vs.HasActivation():
    ComputeActivation()
  if cross_validate:
    experiment.CrossValidateClassifier(vs.exp, vs.layer, learner=algorithm)
  else:
    experiment.TrainAndTestClassifier(vs.exp, vs.layer, learner=algorithm)
  return vs.exp.evaluation[-1].results

def GetFeatures():
  """Get the feature vectors for all images in the experiment."""
  from glimpse.experiment.utils import ExtractFeatures
  vs = _vars()
  if not vs.HasActivation():
    ComputeActivation()
  return ExtractFeatures(vs.layer, vs.exp.extractor.activation)

@docstring.copy_dedent(misc.GetImagePaths)
@depends(CORPUS)
def GetImagePaths():
  return misc.GetImagePaths(GetExperiment())

@docstring.copy_dedent(misc.GetLabelNames)
@depends(CORPUS)
def GetLabelNames():
  return misc.GetLabelNames(GetExperiment())

@docstring.copy_dedent(misc.GetParams)
@depends(MODEL)
def GetParams():
  return misc.GetParams(GetExperiment())

@docstring.copy_dedent(misc.GetNumPrototypes)
@depends(PROTOTYPES)
def GetNumPrototypes(kwidth=0):
  return misc.GetNumPrototypes(GetExperiment(), kwidth)

@docstring.copy_dedent(misc.GetPrototype)
@depends(PROTOTYPES)
def GetPrototype(prototype=0, kwidth=0):
  return misc.GetPrototype(GetExperiment(), prototype, kwidth)

@docstring.copy_dedent(misc.GetImprintLocation)
@depends(PROTOTYPES)
def GetImprintLocation(prototype=0, kwidth=0):
  return misc.GetImprintLocation(GetExperiment(), prototype, kwidth)

@docstring.copy_dedent(misc.GetEvaluationLayers)
@depends(EVALUATION)
def GetEvaluationLayers(evaluation=0):
  return misc.GetEvaluationLayers(GetExperiment(), evaluation)

@docstring.copy_dedent(misc.GetEvaluationResults)
@depends(EVALUATION)
def GetEvaluationResults(evaluation=0):
  return misc.GetEvaluationResults(GetExperiment(), evaluation)

@docstring.copy_dedent(misc.GetPredictions)
@depends(EVALUATION)
def GetPredictions(training=False, evaluation=0):
  return misc.GetPredictions(GetExperiment(), training, evaluation)

@docstring.copy_dedent(misc.ShowS2Activity)
@depends(PROTOTYPES)
def ShowS2Activity(image=0, scale=0, prototype=0, kwidth=0):
  misc.ShowS2Activity(GetExperiment(), image, scale, prototype, kwidth)

@docstring.copy_dedent(misc.ShowPrototype)
@depends(PROTOTYPES)
def ShowPrototype(prototype=0, kwidth=0):
  misc.ShowPrototype(GetExperiment(), prototype, kwidth)

@docstring.copy_dedent(misc.AnnotateImprintedPrototype)
@depends(PROTOTYPES)
def AnnotateImprintedPrototype(prototype=0, kwidth=0):
  misc.AnnotateImprintedPrototype(GetExperiment(), prototype, kwidth)

@docstring.copy_dedent(misc.AnnotateS2Activity)
@depends(PROTOTYPES)
def AnnotateS2Activity(image=0, scale=0, prototype=0, kwidth=0):
  misc.AnnotateS2Activity(GetExperiment(), image, scale, prototype, kwidth)

@docstring.copy_dedent(misc.AnnotateC1Activity)
@depends(MODEL)
def AnnotateC1Activity(image=0, scale=0):
  misc.AnnotateC1Activity(GetExperiment(), image, scale)

@docstring.copy_dedent(misc.AnnotateS1Activity)
@depends(MODEL)
def AnnotateS1Activity(image=0, scale=0):
  misc.AnnotateS1Activity(GetExperiment(), image, scale)

@docstring.copy_dedent(misc.ShowS1Kernels)
@depends(MODEL)
def ShowS1Kernels():
  misc.ShowS1Kernels(GetExperiment())
