"""This module provides a simplified interface for running Glimpse experiments.

The easiest way to use this module is via the top-level functions, such as
:func:`SetCorpus`, which provide a declarative interface in the style of
Matlab(TM). Alternatively, an object-oriented interface is also available by
using the :class:`Experiment` class directly.

"""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import logging
import numpy as np
import os

from .experiment import Experiment
from glimpse import backends
from glimpse.backends import InsufficientSizeException
from glimpse import pools
from glimpse import util
from glimpse.util import docstring
import glimpse.models

__all__ = (
    'ComputeFeatures',
    'CrossValidateSvm',
    'GetExampleCorpus',
    'GetExampleImage',
    'GetExampleImages',
    'GetExperiment',
    'GetImageFeatures',
    'GetLargeExampleCorpus',
    'GetLayer',
    'GetModelClass',
    'GetParams',
    'GetPool',
    'ImprintS2Prototypes',
    'LoadExperiment',
    'MakeHistogramRandomS2Prototypes',
    'MakeModel',
    'MakeNormalRandomS2Prototypes',
    'MakeShuffledRandomS2Prototypes',
    'MakeUniformRandomS2Prototypes',
    'Reset',
    'RunSvm',
    'SetCorpus',
    'SetCorpusSubdirs',
    'SetExperiment',
    'SetLayer',
    'SetModelClass',
    'SetParams',
    'SetPool',
    'SetS2Prototypes',
    'SetTrainTestSplitFromDirs',
    'SetTrainTestSplit',
    'StoreExperiment',
    'TestSvm',
    'TrainSvm',
    'UseCluster',
    'Verbose',
    )

__POOL = None
__MODEL_CLASS = None
__PARAMS = None
__LAYER = None
__EXP = None
__VERBOSE = False

def Reset():
  """Remove the current experiment and revert to default settings."""
  global __EXP, __POOL, __MODEL_CLASS, __PARAMS, __LAYER, __EXP, __VERBOSE
  __EXP = None
  __POOL = None
  __MODEL_CLASS = None
  __PARAMS = None
  __LAYER = None
  __EXP = None
  __VERBOSE = False

Reset()

def SetPool(pool):
  """Set the worker pool used for this experiment."""
  global __POOL
  if util.IsString(pool):
    pool = pool.lower()
    if pool == 'singlecore':
      pool = pools.SinglecorePool()
    elif pool == 'multicore':
      pool = pools.MulticorePool()
    else:
      raise ValueError("Unknown pool type: %s" % pool)
  logging.info("Using pool type: %s", type(pool).__name__)
  __POOL = pool
  return pool

def GetPool():
  """Get the current worker pool used for new experiments."""
  global __POOL
  if __POOL == None:
    __POOL = pools.MakePool()
  return __POOL

def UseCluster(config_file = None, chunksize = None):
  """Use a cluster of worker nodes for any following experiment commands.

  :param str config_file: path to the cluster configuration file

  """
  pkg = pools.GetClusterPackage()
  pool = pkg.MakePool(config_file, chunksize)
  SetPool(pool)

def SetModelClass(model_class = None):
  """Set the model type.

  :param model_class: Model class to use for future experiments. If a name is
     given, it is passed to :func:`GetModelClass
     <glimpse.models.GetModelClass>`.
  :type model_class: class or str

  """
  global __MODEL_CLASS
  if not isinstance(model_class, type):
    model_class = glimpse.models.GetModelClass(model_class)
  logging.info("Using model type: %s", model_class.__name__)
  __MODEL_CLASS = model_class
  return __MODEL_CLASS

def GetModelClass():
  """Get the type that will be used to construct an experiment."""
  if __MODEL_CLASS == None:
    SetModelClass()
  return __MODEL_CLASS

def SetParams(params = None):
  """Set the parameter object that will be used to construct the next
  experiment.

  :param params: Model-specific parameter object to use for future experiments. If
     a filename is given, the parameter object is read from the given file. If no
     parameter object is given, the model's default parameters are used.
  :type params: object or str

  """
  global __PARAMS
  if params == None:
    params = GetModelClass().ParamClass()
  elif isinstance(params, basestring):
    params = util.Load(params)
  __PARAMS = params
  return __PARAMS

def GetParams():
  """Return the parameter object that will be used to construct the next
  experiment.

  """
  if __PARAMS == None:
    SetParams()
  return __PARAMS

def SetLayer(layer = None):
  """Set the layer from which features will be extracted for the next
  experiment.

  :param layer: Layer from which to compute feature vectors.
  :type layer: str or :class:`glimpse.models.misc.LayerSpec`

  """
  global __LAYER
  if layer == None:
    layer = GetModelClass().LayerClass.TopLayer()
  elif isinstance(layer, str):
    layer = GetModelClass().LayerClass.FromName(layer)
  __LAYER = layer
  return __LAYER

def GetLayer():
  """Return the layer from which features will be extracted for the next
  experiment.

  """
  if __LAYER == None:
    SetLayer()
  return __LAYER

def MakeModel(params = None):
  """Create a Glimpse model.

  .. seealso::
     :func:`SetModelClass`

  """
  if params == None:
    params = GetParams()
  return __MODEL_CLASS(backends.MakeBackend(), params)

def GetExperiment():
  """Get the current experiment object.

  :rtype: :class:`Experiment`

  """
  if __EXP == None:
    SetExperiment()
  return __EXP

def SetExperiment(model = None, layer = None):
  """Manually create a new experiment.

  This function generally is not called directly. Instead, an experiment object
  is implicitly created when needed.

  :param model: The Glimpse model to use for processing images.
  :param layer: The layer activity to use for features vectors.
  :type layer: :class:`LayerSpec <glimpse.models.misc.LayerSpec>` or str
  :returns: The new experiment object.
  :rtype: :class:`Experiment`

  .. seealso::
    :func:`SetPool`, :func:`SetParams`, :func:`SetModelClass`, :func:`SetLayer`.

  """
  global __EXP
  if model == None:
    model = MakeModel()
  if layer == None:
    layer = GetLayer()
  elif isinstance(layer, str):
    layer = model.LayerClass.FromName(layer)
  __EXP = Experiment(model, layer, pool = GetPool())
  return __EXP

@docstring.copy_dedent(Experiment.ImprintS2Prototypes)
def ImprintS2Prototypes(num_prototypes):
  """" """
  if __VERBOSE:
    print "Imprinting %d prototypes" % num_prototypes
  result = GetExperiment().ImprintS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

@docstring.copy_dedent(Experiment.MakeUniformRandomS2Prototypes)
def MakeUniformRandomS2Prototypes(num_prototypes, low = None, high = None):
  """" """
  if __VERBOSE:
    print "Making %d uniform random prototypes" % num_prototypes
  result = GetExperiment().MakeUniformRandomS2Prototypes(num_prototypes, low,
      high)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

@docstring.copy_dedent(Experiment.MakeShuffledRandomS2Prototypes)
def MakeShuffledRandomS2Prototypes(num_prototypes):
  """" """
  if __VERBOSE:
    print "Making %d shuffled random prototypes" % num_prototypes
  result = GetExperiment().MakeShuffledRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

@docstring.copy_dedent(Experiment.MakeHistogramRandomS2Prototypes)
def MakeHistogramRandomS2Prototypes(num_prototypes):
  """" """
  if __VERBOSE:
    print "Making %d histogram random prototypes" % num_prototypes
  result = GetExperiment().MakeHistogramRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

@docstring.copy_dedent(Experiment.MakeNormalRandomS2Prototypes)
def MakeNormalRandomS2Prototypes(num_prototypes):
  """" """
  if __VERBOSE:
    print "Making %d normal random prototypes" % num_prototypes
  result = GetExperiment().MakeNormalRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def SetS2Prototypes(prototypes):
  """Set the S2 prototypes from an array or a file.

  :param prototypes: The set of prototypes or a path to a file containing the
     prototypes.
  :type prototypes: str or ndarray of float

  """
  if isinstance(prototypes, basestring):
    prototypes = util.Load(prototypes)
  elif not (isinstance(prototypes, np.ndarray) or isinstance(prototypes, list) \
    or isinstance(prototypes, tuple)):
    raise ValueError("Please specify an array of prototypes, or the path to a "
        "file.")
  GetExperiment().SetS2Prototypes(prototypes)

@docstring.copy_dedent(Experiment.SetCorpus)
def SetCorpus(corpus_dir, classes = None, balance = False):
  """" """
  return GetExperiment().SetCorpus(corpus_dir, classes = classes,
      balance = balance)

@docstring.copy_dedent(Experiment.SetCorpusSubdirs)
def SetCorpusSubdirs(corpus_subdirs, classes = None, balance = False):
  """" """
  return GetExperiment().SetCorpusSubdirs(corpus_subdirs, classes = classes,
      balance = balance)

@docstring.copy_dedent(Experiment.SetTrainTestSplitFromDirs)
def SetTrainTestSplitFromDirs(train_dir, test_dir, classes = None):
  """" """
  return GetExperiment().SetTrainTestSplitFromDirs(train_dir, test_dir, classes)

@docstring.copy_dedent(Experiment.SetTrainTestSplit)
def SetTrainTestSplit(train_images, test_images, classes):
  """" """
  return GetExperiment().SetTrainTestSplit(train_images, test_images, classes)

@docstring.copy_dedent(Experiment.ComputeFeatures)
def ComputeFeatures(raw = False):
  """" """
  GetExperiment().ComputeFeatures(raw = raw)

@docstring.copy_dedent(Experiment.GetImageFeatures)
def GetImageFeatures(images, resize = None, raw = False, save_all = False,
    block = True):
  """ """
  if isinstance(images, basestring):
    images = [ images ]  # Single image was passed, wrap it in a list.
    single_input = True
  else:
    single_input = False
  results = GetExperiment().GetImageFeatures(images, resize = resize, raw = raw,
      save_all = save_all, block = block)
  if single_input:
    results = results[0]
  return results

@docstring.copy_dedent(Experiment.CrossValidateSvm)
def CrossValidateSvm():
  """" """
  return GetExperiment().CrossValidateSvm()

@docstring.copy_dedent(Experiment.TrainSvm)
def TrainSvm():
  """" """
  return GetExperiment().TrainSvm()

@docstring.copy_dedent(Experiment.TestSvm)
def TestSvm():
  """" """
  return GetExperiment().TestSvm()

@docstring.copy_dedent(Experiment.RunSvm)
def RunSvm(cross_validate = False):
  """" """
  exp = GetExperiment()
  if cross_validate:
    if __VERBOSE:
      print "Computing cross-validated SVM performance on %d images" % \
          sum(map(len, exp.GetImages()))
  else:
    if __VERBOSE:
      print "Train SVM on %d images" % sum(map(len, exp.train_images))
      print "  and testing on %d images" % sum(map(len, exp.test_images))
  train_accuracy, test_accuracy = exp.RunSvm(cross_validate)
  if __VERBOSE:
    print "  done: %s s" % exp.svm_time
    print "Time to compute feature vectors: %s s" % \
        exp.compute_feature_time
    print "Accuracy is %.3f on training set, and %.3f on test set." % \
        (train_accuracy, test_accuracy)
  return train_accuracy, test_accuracy

@docstring.copy_dedent(Experiment.Store)
def StoreExperiment(root_path):
  """" """
  return GetExperiment().Store(root_path)

@docstring.copy_dedent(Experiment.Load)
def LoadExperiment(root_path):
  """" """
  global __EXP
  __EXP = Experiment.Load(root_path)
  __EXP.pool = GetPool()
  return __EXP

def Verbose(flag = True):
  """Enable (or disable) logging.

  :param bool flag: True if logging should be used, else False.

  """
  global __VERBOSE
  __VERBOSE = flag

def GetExampleCorpus():
  """Get an example image corpus.

  :returns: Corpus path.
  :rtype: str

  """
  return os.path.join(os.path.dirname(__file__), 'data', 'small-corpus')

def GetLargeExampleCorpus():
  """Get a larger example image corpus.

  :returns: Corpus path.
  :rtype: str

  """
  return os.path.join(os.path.dirname(__file__), 'data', 'large-corpus')

def GetExampleImage():
  """Get a single example image.

  :returns: Image path.
  :rtype: str

  """
  return os.path.join(GetExampleCorpus(), 'cats', 'Marcus_bed.jpg')

def GetExampleImages():
  """Get multiple example images.

  :returns: Image paths.
  :rtype: list of str

  """
  corpus = GetExampleCorpus()
  return [ os.path.join(corpus, 'cats', 'Marcus_bed.jpg'), os.path.join(corpus,
      'dogs', '41-27Monate1.JPG') ]
