# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

# Purpose: make the common case (imprint prototypes and train/test svm) easy,
# while also supporting ad hoc data analysis.
# As such, the experiment object encapsulates all generated data for an
# experiment, without assuming a fixed progression of steps in the experimental
# protocol. Instead, operations use call-time validation of required data, and
# throw specific exceptions when dependencies are missing.
# NOTE: The GLAB interface assumes the use of the default Glimpse model.

import logging
import numpy as np
import os
from pprint import pformat

from glimpse.util.data import Data
from .utils import (TimedLogging, DirReader, ReadCorpusDirs, BalanceCorpus,
    ExtractFeatures)
from glimpse.models.base import BuildLayer
from glimpse.pools import MakePool
from .prototype_algorithms import *
from glimpse.util.callback import Callback
from glimpse.util.learn import (ChooseTrainingSet, FitClassifier,
    ScoreClassifier, ResolveScoreFunction, ResolveLearner)
from glimpse.util import learn

class CorpusData(Data):
  #: (list of str) Path for each input image
  paths = None
  #: (1D ndarray of int) Class label for each input image
  labels = None
  #: (list of str) Name for each object class, with indices corresponding to
  #: the values in the `labels` attribute.
  class_names = None
  #: (1D ndarray of bool) Mask indicating if each image is in the training set,
  #: as specified by user.
  training_set = None

class ExtractorData(Data):
  #: (Model) Glimpse model
  model = None
  #: (Params) Parameters for Glimpse model
  #: XXX is this needed?
  params = None
  #: (1D ndarray of bool) Mask indicating if each image is in the training set,
  #: as used for prototype learning. This is `None` if corpus.training_set is
  #: given, or if prototypes are not derived from training data.
  training_set = None
  #: (tuple of BaseState) Activation maps for each image, organized by (image,
  #: layer, scale, y, x).
  activation = None

class EvaluationData(Data):
  """Parameters and results for classifier evaluation.

  The evaluator creates feature vectors on the fly from activation maps in
  extractor.activation, based on values in evaluation.layers list. Thus, the set
  of feature vectors are not explicitly stored. The method for building these
  features is specified by the user via the ``feature_builder`` argument. See
  :func:`TrainAndTestClassifier` and :func:`CrossValidateClassifier`.

  """
  #: Layers from which features are extracted.
  layers = None
  #: (1D ndarray of bool) Mask indicating if each image is in the training set,
  #: as used during evaluation. This is `None` if either corpus.training_set or
  #: extractor.training_set is given, or if a fixed training set is not used
  #: during evaluation (e.g., if cross-validation is used).
  training_set = None
  #: Outcome of evaluation. This may contain one or more measurements of
  #: classifier accuracy, AUC, etc.
  results = Data()

class ExperimentData(Data):
  """Results and settings for an experiment."""

  #: Input image data (exactly one).
  corpus = CorpusData()
  #: Feature extraction (exactly one).
  extractor = ExtractorData()
  #: (list of EvaluationData) Classification performance on feature data (zero
  #: or more).
  evaluation = list()

class ExpError(Exception):
  """Indicates that an error occurred while processing an Experiment."""
  pass

##############################################################
# The following functions operate on ExperimentData objects. #
##############################################################

def ResolveLayers(exp, layers):
  """Resolve layer names to LayerSpec objects.

  This is an internal function.

  :type layers: str or list of str
  :param layers: One or more model layers to compute.
  :rtype: list of :class:`LayerSpec`
  :return: Resolved layer specs.

  """
  if exp.extractor.model is None:
    raise ValueError("Experiment object has no model")
  if layers is None:
    raise ValueError("Layer information must be non-empty")
  if not hasattr(layers, '__len__') or isinstance(layers, basestring):
    layers = (layers,)
  out = list()
  for l in layers:
    if isinstance(l, basestring):
      l = exp.extractor.model.LayerClass.FromName(l)
    out.append(l)
  return out

def Verbose(flag=None):
  """Set the verbosity of log output.

  :param bool flag: Whether to enable verbose logging.

  """
  if flag is None:
    flag = os.environ.get('GLIMPSE_VERBOSE')
  if isinstance(flag, basestring):
    flag = flag.lower() in ('1', 'true')
  if flag:
    level = logging.INFO
  else:
    level = logging.ERROR
  logging.getLogger().setLevel(level)
  return level

def SetModel(exp, model=None, params=None, **kw):
  """Add a model to an existing experiment.

  If `model` is passed, it is stored in the experiment's `extractor.model`
  attribute. Otherwise, the parameters are created/updated, and used to create a
  new :class:`Model` object.

  :type model: :class:`Model`
  :param model: Existing model object to use.
  :type params: :class:`Params`
  :param params: Existing model parameters to use.

  All remaining keyword arguments are treated as model parameters and overwrite
  values in the (created or passed) `params` argument. This allows the set of
  parameters to be specified in short-hand as

  >>> SetModel(param1=1, param2=2)

  without creating a :class:`Params` object.

  """
  from glimpse.models.ml import Model, Params
  if model is None:
    if params is None:
      params = Params()
    for k,v in kw.items():
      setattr(params, k, v)
    model = Model(params)
  exp.extractor.model = model

### CORPUS ###

def SetCorpus(exp, root_dir, balance=False, reader=None):
  """Read images from the corpus directory.

  This function assumes that each sub-directory contains images for exactly one
  object class, with a different object class for each sub-directory. Training
  and testing subsets are chosen automatically.

  :param str root_dir: Path to corpus directory.
  :param bool balance: Ensure an equal number of images from each class (by
     random selection).
  :param reader: Filesystem reader.

  .. seealso::
     :func:`SetCorpusSubdirs`, :func:`SetCorpusSplit`

  """
  logging.info("Reading class sub-directories from: %s", root_dir)
  if reader is None:
    reader = DirReader(ignore_hidden=True)
  subdirs = reader.ReadDirs(root_dir)
  subdirs.sort()
  if not subdirs:
    raise ExpError("Corpus directory is empty: %s" % root_dir)
  SetCorpusSubdirs(exp, subdirs, balance, reader)

def SetCorpusSubdirs(exp, subdirs, balance=False, reader=None):
  """Read images from per-class corpus sub-directories.

  This function assumes that each sub-directory contains images for exactly one
  object class, with a different object class for each sub-directory. Training
  and testing subsets are chosen automatically.

  :type subdirs: iterable of str
  :param subdirs: Path of each corpus sub-directory.
  :param bool balance: Ensure an equal number of images from each class (by
     random selection).
  :param reader: Filesystem reader.

  .. seealso::
     :func:`SetCorpus`, :func:`SetCorpusSplit`

  """
  logging.info("Reading images from class directories: %s", pformat(subdirs))
  if reader is None:
    reader = DirReader(ignore_hidden=True)
  paths, labels = ReadCorpusDirs(subdirs, reader)
  if balance:
    logging.info("Balancing number of images per class")
    mask = BalanceCorpus(labels, shuffle=True)
    # get subset of image paths and corresponding labels
    paths = map(paths.__getitem__, np.where(mask)[0])
    labels = labels[mask]
  exp.corpus.paths = paths
  exp.corpus.labels = labels
  exp.corpus.class_names = np.array(map(os.path.basename, subdirs))

def SetCorpusSplit(exp, train_dir, test_dir, reader=None):
  """Read images and training information from the corpus directory.

  This function assumes that the `train_dir` and `test_dir` have the same set of
  sub-directories. Each sub-directory shoudl contain images for exactly one
  object class, with a different object class for each sub-directory.

  :param str train_dir: Path to corpus of training images.
  :param str test_dir: Path to corpus of test images.
  :param reader: Filesystem reader.

  .. seealso::
     :func:`SetCorpus`, :func:`SetCorpusSubdirs`

  """
  logging.info("Reading training set sub-directories from: %s, "
      "and test set sub-directories from %s", train_dir, test_dir)
  if reader is None:
    reader = DirReader(ignore_hidden=True)
  train_subdirs = sorted(reader.ReadDirs(train_dir))
  test_subdirs = sorted(reader.ReadDirs(test_dir))
  if map(os.path.basename, train_subdirs) != map(os.path.basename, test_subdirs):
    raise ExpError("Class subdirectories for training and testing must match")
  train_paths, train_labels = ReadCorpusDirs(train_subdirs, reader)
  test_paths, test_labels = ReadCorpusDirs(test_subdirs, reader)
  training_set = np.zeros((len(train_paths) + len(test_paths),), np.bool)
  training_set[:len(train_paths)] = True
  exp.corpus.training_set = training_set
  exp.corpus.paths = np.hstack((train_paths, test_paths))
  exp.corpus.labels = np.hstack((train_labels, test_labels))
  exp.corpus.class_names = np.array(map(os.path.basename, train_subdirs))

### EXTRACTION ###

def _MakeTrainingExp(exp, train_size=None):
  if train_size is None:
    train_size = 0.5
  # Create a sub-experiment containing only training data.
  if exp.corpus.training_set is not None:
    training_set = exp.corpus.training_set
  else:
    if exp.extractor.training_set is None:
      # This could happen if make_training_exp is called repeatedly.
      exp.extractor.training_set = ChooseTrainingSet(exp.corpus.labels,
          train_size)
    training_set = exp.extractor.training_set
  training_exp = ExperimentData()
  training_exp.corpus.paths = [exp.corpus.paths[i]
      for i in np.where(training_set)[0]]
  training_exp.corpus.labels = exp.corpus.labels[training_set]
  training_exp.corpus.training_set = np.ones(len(training_exp.corpus.paths),
      bool)
  training_exp.extractor.model = exp.extractor.model
  return training_exp

def MakePrototypes(exp, num_prototypes, algorithm, pool=None, train_size=None,
    progress=None):
  """Create a set of S2 prototypes, and use them for this experiment.

  :param int num_prototypes: Number of prototypes to create.
  :type algorithm: callable or str
  :param algorithm: Prototype learning algorithm, or name of such an algorithm.
  :param pool: Worker pool to use for parallel computation.
  :type train_size: float or int
  :param train_size: Size of training split, specified as a fraction
     (between 0 and 1) of total instances or as a number of instances (1 to N,
     where N is the number of available instances).
  :param progress: Handler for incremental progress updates.

  If `algorithm` is a function, it will be called as

  >>> algorithm(num_prototypes, model, make_training_exp, pool, progress)

  where `model` is the experiment's hierarchical model. The argument
  `make_training_exp` is a function that takes no arguments and returns the
  training experiment (i.e., an :class:`ExperimentData` object containing only
  training images and labels. Calling this method has the side-effect that the
  original experiment's `extractor.training_set` attribute will be set if
  `corpus.training_set` is empty.

  If `algorithm` has a `locations` attribute, it is assumed to contain a list of
  `num_prototypes` x 4 arrays with list indexed by kernel width and array
  indexed by prototype. Each row of the array contains (image index, scale
  offset, y-offset, x-offset). The algorithm sets image indices relative to the
  training set, and this function rewrites those indices relative to the full
  corpus.

  """
  # Note that if `algorithm` does not call this function, then
  # `extractor.training_set` will never be constructed.
  make_training_exp = Callback(_MakeTrainingExp, exp, train_size)
  if isinstance(algorithm, basestring):
    algorithm = ResolveAlgorithm(algorithm)
    algorithm = algorithm()  # create algorithm object of given class
  if pool is None:
    pool = MakePool()
  model = exp.extractor.model
  with TimedLogging(logging.INFO, "Learning prototypes"):
    protos = algorithm(num_prototypes, model, make_training_exp, pool, progress)
  for p in protos:
    if len(p) != num_prototypes:
      raise ExpError("Prototype learning algorithm returned wrong number of"
          " prototypes")
  # Special case: fix up imprint locations. User expects image indices to be
  # given relative to exp.corpus.paths, but they are currently given relative to
  # the training set.
  if hasattr(algorithm, 'locations') and algorithm.locations is not None:
    if exp.corpus.training_set is not None:
      training_indices = np.where(exp.corpus.training_set)[0]
    else:
      assert exp.extractor.training_set is not None
      training_indices = np.where(exp.extractor.training_set)[0]
    for locs in algorithm.locations:
      locs[:,0] = training_indices[locs[:,0]]
  exp.extractor['prototype_algorithm'] = algorithm
  model.s2_kernels = protos

def _ComputeActivation(model, layers, images, pool, save_all=False,
    progress=None):
  LC = model.LayerClass
  for l in layers:
    if LC.IsSublayer(LC.C1, l) and model.s2_kernels is None:
      raise ExpError("Need S2 kernels to compute %s layer activity, " % l +
          "but none were specified")
  logging.info("Computing %s activation maps for %d images",
      "/".join(l.name for l in layers), len(images))
  images = map(model.MakeState, images)
  builder = Callback(BuildLayer, model, layers, save_all=save_all)
  with TimedLogging(logging.INFO, "Computing activation maps"):
    # Compute model states containing desired features.
    return pool.map(builder, images, progress=progress)

def ComputeActivation(exp, layers, pool, save_all=False, progress=None):
  """Compute the model activity for all images in the experiment.

  :type layers: str or list of str
  :param layers: One or more model layers to compute.
  :param pool: Worker pool to use for parallel computation.
  :param bool save_all: Whether to save activation for all model layers, rather
     than just those in `layers`.
  :param progress: Handler for incremental progress updates.

  """
  model = exp.extractor.model
  layers = ResolveLayers(exp, layers)
  exp.extractor.activation = _ComputeActivation(model, layers, exp.corpus.paths,
      pool, save_all, progress)

def GetImageFeatures(exp, layers, images, pool, feature_builder=None,
    progress=None):
  """Get feature vectors for a set of images outside the experiment's corpus.

  :type layers: str or list of str
  :param layers: One or more model layers to compute.
  :type images: list of str
  :param images: Path for images from which to extract features.
  :param pool: Worker pool to use for parallel computation.
  :param callable feature_builder: A feature builder, such as
     :func:`ExtractFeatures` or :func:`ExtractHistogramFeatures`.
  :param progress: Handler for incremental progress updates.

  This function does not affect the experiment object.

  """
  model = exp.extractor.model
  layers = ResolveLayers(exp, layers)
  states = _ComputeActivation(model, layers, images, pool, save_all=False,
      progress=progress)
  if feature_builder is None:
    feature_builder = ExtractFeatures
  features = feature_builder(layers, states)
  return features


### EVALUATION ###

def TrainAndTestClassifier(exp, layers, learner=None, train_size=None,
    feature_builder=None, score_func=None):
  """Evaluate extracted features using a fixed train/test split.

  :type layers: str or list of str
  :param layers: Layers of model activity from which to extract features.
  :param learner: Learning algorithm, which is fit to features. This should be a
     scikit-learn classifier object. If not set, a LinearSVC object is used.
  :type train_size: float or int
  :param train_size: Size of training split, specified as a fraction
     (between 0 and 1) of total instances or as a number of instances (1 to N,
     where N is the number of available instances).
  :param callable feature_builder: A feature builder, such as
     :func:`ExtractFeatures` or :func:`ExtractHistogramFeatures`.
  :param str score_func: Name of the scoring function to use, as specified by
     :func:`ResolveScoreFunction`.

  Creates a new entry in the experiment's `evaluation` list, and sets the
  `feature_builder`, `classifier`, `training_predictions`, `training_score`,
  `score_func`, `predictions`, and `score` keys in its `results` dictionary.

  """
  if exp.extractor.activation is None:
    raise Exception("Must compute layer activation before training classifier")
  layer_names = layers
  layers = ResolveLayers(exp, layers)
  if feature_builder is None:
    feature_builder = ExtractFeatures
  evaluation = EvaluationData(layers = layers)
  evaluation.results['feature_builder'] = feature_builder.func_name
  features = feature_builder(layers, exp.extractor.activation)
  logging.info("Evaluating classifier on fixed train/test split " +
      ("on %d images using %d features " % features.shape) +
      ("from layer(s): %s" % layer_names))
  features = features.astype(float)  # classifier assumes float data
  labels = exp.corpus.labels
  if exp.extractor.training_set is not None:
    training_set = exp.extractor.training_set
  elif exp.corpus.training_set is not None:
    training_set = exp.corpus.training_set
  else:
    if train_size is None:
      train_size = 0.5
    evaluation.training_set = ChooseTrainingSet(exp.corpus.labels,
        train_size)
    training_set = evaluation.training_set
  with TimedLogging(logging.INFO, "Training on %d images" % training_set.sum()):
    clf = FitClassifier(features[training_set], labels[training_set],
        algorithm=learner)
  logging.info("Classifier is %s" % clf)
  evaluation.results['classifier'] = clf
  score_func = ResolveScoreFunction(score_func)
  evaluation.results['score_func'] = score_func
  with TimedLogging(logging.INFO, "Scoring on training set (%d images)" %
      training_set.sum()):
    score,predictions = ScoreClassifier(features[training_set],
        labels[training_set], clf=clf, score_func=score_func)
    logging.info("Classifier %s on training set is %f" % (score_func, score))
    evaluation.results['training_predictions'] = predictions
    evaluation.results['training_score'] = score
  if training_set.sum() < labels.size:
    with TimedLogging(logging.INFO, "Scoring on testing set (%d images)" %
        (len(training_set) - training_set.sum())):
      score,predictions = ScoreClassifier(features[~training_set],
          labels[~training_set], clf=clf, score_func=score_func)
    logging.info("Classifier %s on test set is %f" % (score_func, score))
    evaluation.results['predictions'] = predictions
    evaluation.results['score'] = score  # classification score on test set
  exp.evaluation.append(evaluation)

def CrossValidateClassifier(exp, layers, learner=None, feature_builder=None,
    num_folds=10, score_func=None):
  """Evaluate extracted features using a fixed train/test split.

  :type layers: str or list of str
  :param layers: Layers of model activity from which to extract features.
  :param learner: Learning algorithm, which is fit to features. This should be a
     scikit-learn classifier object. If not set, a LinearSVC object is used.
  :param callable feature_builder: A feature builder, such as
     :func:`ExtractFeatures` or :func:`ExtractHistogramFeatures`.
  :param int num_folds: Number of folds to use for cross-validation.
  :param str score_func: Name of the scoring function to use, as specified by
     :func:`ResolveScoreFunction`.

  Creates a new entry in the experiment's `evaluation` list, and sets the
  `feature_builder`, `cross_validate`, `crossval_learner`, `score_func`, and
  `score` keys in its `results` dictionary.

  """
  layer_names = layers
  layers = ResolveLayers(exp, layers)
  if feature_builder is None:
    feature_builder = ExtractFeatures
  evaluation = EvaluationData(layers = layers)
  evaluation.results['feature_builder'] = feature_builder.func_name
  features = feature_builder(layers, exp.extractor.activation)
  logging.info(("Evaluating classifier via %d-fold " % num_folds) +
      ("cross-validation on %d images " % features.shape[0]) +
      ("using %s features " % features.shape[1]) +
      ("from layer(s): %s" % layer_names))
  features = features.astype(float)  # classifier assumes float data
  labels = exp.corpus.labels
  if len(labels) < num_folds:
    raise ExpError("Need at least %d images to use %d-way cross-validation" %
        (num_folds, num_folds))
  if exp.corpus.training_set is not None:
    raise ExpError("Cross-validation is unavailable when a fixed training set "
        "is specified")
  elif exp.extractor.training_set is not None:
    raise ExpError("Cross-validation is unavailable when a fixed training set"
        " is used for prototype learning")
  evaluation.results['cross_validate'] = True
  learner = ResolveLearner(learner)
  evaluation.results['crossval_learner'] = learner  # algorithm with parameters
  with TimedLogging(logging.INFO, "Cross-validation"):
    score = learn.CrossValidateClassifier(features, labels, num_folds=num_folds,
        algorithm=learner)
    logging.info("Mean classifier accuracy is %.1f%%" % np.array(score).mean())
  evaluation.results['score'] = score
  exp.evaluation.append(evaluation)
