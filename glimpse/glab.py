#!/usr/bin/python

"""The glab module provides a command-driven, matlab-like interface to the
Glimpse project.

The following is an example of a basic experiment.

from glimpse.glab import *

# Read images from the directory "indir/corpus", which is assumed to have two
# sub-directories (one for each class). The names of the sub-directories
# corresponds to the class names. The first sub-directory (as given by the
# os.listdir() method) is the "positive" class, while the second is the
# "negative" class. Half of the images from each class are randomly selected to
# form the training set, while the other half is held for testing.
SetCorpus("indir/corpus")
# Imprint a set of 10 S2 prototypes from the training set.
ImprintS2Prototypes(10)
# Build and test an SVM classifier based on IT activity.
TrainSvm()
TestSvm()
# Store the configuration and results of the experiment to the file
# "outdir/exp.dat". Additionally, store the SVM classifier to the file
# "outdir/exp.dat.svm".
StoreExperiment("outdir/exp.dat")

# By default, the order of the classes is somewhat arbitrary. Instead, we could
# specify that "class1" is the positive SVM class, while "class2" is the
# negative SVM class,
SetCorpus('corpus', classes = ('cls1', 'cls2'))

# If we had wanted to build SVM feature vectors from C1 activity, instead of IT
# activity, we could have initialized the experiment before setting the corpus.
SetExperiment(layer = 'C1')
SetCorpus("indir/corpus")

# If we had wanted to configure the parameters of the Glimpse model, we could
# have constructed the model manually when initializing the experiment.
params = Params()
params.num_scales = 8
SetExperiment(model = Model(params))
SetCorpus("indir/corpus")
"""

from glimpse import backends
from glimpse.models import viz2
from glimpse import pools, util
import itertools
import numpy as np
import operator
import os
import sys
import time

__all__ = ( 'UseCluster', 'SetModelClass', 'MakeParams', 'MakeModel',
    'GetExperiment', 'SetExperiment', 'ImprintS2Prototypes',
    'MakeRandomPrototypes', 'SetCorpus', 'SetTrainTestSplit', 'TrainSvm',
    'TestSvm', 'LoadExperiment', 'StoreExperiment', 'Verbose')

def ChainLists(*iterables):
  """Concatenate several sequences to form a single list."""
  return list(itertools.chain(*iterables))

class Experiment(object):

  def __init__(self, model, layer, pool = None, ordered = True):
    """Create a new experiment.
    model -- the Glimpse model to use for processing images
    layer -- (LayerSpec or str) the layer activity to use for features vectors
    pool -- a serializable worker pool
    ordered -- (bool) whether the order of SVM feature vectors must be preserved
               within each class
    """
    if model == None:
      model = Model()
    if layer == None:
      layer = model.Layer.IT
    elif isinstance(layer, str):
      layer = model.Layer.FromName(layer)
    if pool == None:
      pool = pools.MakePool()
    self.model = model
    self.pool = pool
    self.layer = layer
    self.ordered = ordered
    # Initialize attributes used by an experiment
    self.classes = []
    self.classifier = None
    self.corpus = None
    self.prototype_source = None
    self.train_images = None
    self.test_images = None
    self.train_test_split = None
    self.pos_features = None
    self.neg_features = None
    self.train_error = None
    self.test_error = None
    self.prototype_construction_time = None
    self.svm_train_time = None
    self.svm_test_time = None

  @property
  def s2_prototypes(self):
    return self.model.s2_kernels

  @s2_prototypes.setter
  def s2_prototypes(self, value):
    self.prototype_source = 'manual'
    self.model.s2_kernels = value

  def __str__(self):
    values = dict(self.__dict__)
    values['classes'] = ", ".join(values['classes'])
    return """Experiment:
  corpus: %(corpus)s
  classes: %(classes)s
  train_test_split: %(train_test_split)s
  model: %(model)s
  layer: %(layer)s
  prototype_source: %(prototype_source)s
  train_error: %(train_error)s
  test_error: %(test_error)s""" % values

  __repr__ = __str__

  def ImprintS2Prototypes(self, num_prototypes):
    """Imprint a set of S2 prototypes from a set of training images.
    num_prototypes -- (int) the number of C1 patches to sample
    """
    if self.train_images == None:
      sys.exit("Please specify the training corpus before calling "
          "ImprintS2Prototypes().")
    start_time = time.time()
    image_files = ChainLists(*self.train_images)
    # Represent each image file as an empty model state.
    input_states = map(self.model.MakeStateFromFilename, image_files)
    prototypes, _ = self.model.ImprintS2Prototypes(num_prototypes,
        input_states, normalize = True, pool = self.pool)
    # Store new prototypes in model.
    self.prototype_source = 'imprinted'
    self.model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time

  def MakeRandomPrototypes(self, num_prototypes):
    """Create a set of S2 prototypes with uniformly random entries.
    num_prototypes -- (int) the number of S2 prototype arrays to create
    """
    start_time = time.time()
    shape = (num_prototypes,) + tuple(self.model.s2_kernel_shape)
    prototypes = np.random.uniform(0, 1, shape)
    for p in prototypes:
      p /= np.linalg.norm(p)
    self.model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time

  def ComputeFeatures(self, input_states):
    """Return the activity of the model's output layer for a set of images.
    input_states -- (State iterable) model states containing image data
    RETURN (iterable) a feature vector for each image
    """
    L = self.model.Layer
    if self.layer in (L.S2, L.C2, L.IT) and self.model.s2_kernels == None:
      sys.exit("Please set the S2 prototypes before computing feature vectors "
          "for layer %s." % self.layer.name)
    builder = self.model.BuildLayerCallback(self.layer, save_all = False)
    # Compute model states containing IT features.
    if self.ordered:
      output_states = self.pool.map(builder, input_states)
    else:
      output_states = self.pool.imap_unordered(builder, input_states)
    # Look up the activity values for the output layer, and convert them all to
    # a single vector.
    return [ util.ArrayListToVector(state[self.layer.id])
        for state in output_states ]

  def _ReadCorpusDir(self, corpus_dir, classes = None):
    if classes == None:
      classes = os.listdir(corpus_dir)
    try:
      def ReadClass(cls):
        class_dir = os.path.join(corpus_dir, cls)
        return [ os.path.join(class_dir, img) for img in os.listdir(class_dir) ]
      return map(ReadClass, classes)
    except OSError, e:
      sys.exit("Failed to read corpus directory: %s" % e)

  def SetCorpus(self, corpus_dir, classes = None):
    """Read images from the corpus directory, and choose training and testing
    subsets automatically. Use this instead of SetTrainTestSplit().
    corpus_dir -- (str) path to corpus directory
    classes -- (list) set of class names. Use this to ensure a given order to
               the SVM classes. When applying a binary SVM, the first class is
               treated as positive and the second class is treated as negative.
    """
    if classes == None:
      classes = os.listdir(corpus_dir)
    self.classes = classes
    #~ corpus_dir = os.path.abspath(corpus_dir)
    self.corpus = corpus_dir
    self.train_test_split = 'automatic'
    images_per_class = self._ReadCorpusDir(corpus_dir, classes)
    # Randomly reorder image lists.
    for images in images_per_class:
      np.random.shuffle(images)
    # Use first half of images for training, and second half for testing.
    self.train_images = [ images[ : len(images)/2 ]
        for images in images_per_class ]
    self.test_images = [ images[ len(images)/2 : ]
        for images in images_per_class ]

  def SetTrainTestSplit(self, train_dir, test_dir, classes = None):
    """Read images from the corpus directories, setting the training and testing
    subsets manually. Use this instead of SetCorpus()."""
    if classes == None:
      classes = os.listdir(train_dir)
    self.classes = classes
    #~ train_dir = os.path.abspath(train_dir)
    #~ test_dir = os.path.abspath(test_dir)
    self.corpus = (train_dir, test_dir)
    self.train_test_split = 'manual'
    self.train_images = self._ReadCorpusDir(train_dir, classes)
    self.test_images = self._ReadCorpusDir(test_dir, classes)

  def _ComputeLibSvmFeatures(self, pos_images, neg_images):
    """Internal helper function for TrainSvm() and TestSvm()."""
    # Compute features for images in the positive class.
    input_states = map(self.model.MakeStateFromFilename, pos_images)
    self.pos_features = self.ComputeFeatures(input_states)
    # Compute features for images in the positive class.
    input_states = map(self.model.MakeStateFromFilename, neg_images)
    self.neg_features = self.ComputeFeatures(input_states)
    classes = [1] * len(self.pos_features) + [-1] * len(self.neg_features)
    features = self.pos_features + self.neg_features
    # Convert feature vectors from np.array to list objects
    features = map(list, features)
    return classes, features

  def TrainSvm(self):
    """Train an SVM classifier from the set of training images."""
    if self.train_images == None:
      sys.exit("Please specify the training corpus before calling TrainSvm().")
    start_time = time.time()
    classes, features = self._ComputeLibSvmFeatures(*self.train_images)
    options = '-q'  # don't write to stdout
    # Use delayed import of LIBSVM library, so non-SVM methods are always
    # available.
    import svmutil
    self.classifier = svmutil.svm_train(classes, features, options)
    options = ''  # can't disable writing to stdout
    predicted_labels, acc, decision_values = svmutil.svm_predict(classes,
        features, self.classifier, options)
    self.train_error = 1. - float(acc[0]) / 100.
    self.svm_train_time = time.time() - start_time
    return self.train_error

  def TestSvm(self):
    """Apply the classifier to a set of test images.
    RETURN (float) accuracy
    """
    if self.classifier == None:
      sys.exit("Please train the classifier before calling TestSvm().")
    if self.test_images == None:
      sys.exit("Please specify the testing corpus before calling TestSvm().")
    start_time = time.time()
    classes, features = self._ComputeLibSvmFeatures(*self.test_images)
    options = ''  # can't disable writing to stdout
    # Use delayed import of LIBSVM library, so non-SVM methods are always
    # available.
    import svmutil
    predicted_labels, acc, decision_values = svmutil.svm_predict(classes,
        features, self.classifier, options)
    # Ignore mean-squared error and correlation coefficient
    self.test_error = 1. - float(acc[0]) / 100.
    self.svm_test_time = time.time() - start_time
    return self.test_error

  def Store(self, root_path):
    """Save the experiment to disk."""
    # We modify the value of the "classifier" attribute, so cache it.
    classifier = self.classifier
    pool = self.pool
    self.pool = None  # can't serialize some pools
    # Use "classifier" attribute to indicate whether LIBSVM classifier is
    # present.
    if classifier != None:
      self.classifier = True
    util.Store(self, root_path)
    if classifier != None:
      # Use delayed import of LIBSVM library, so non-SVM methods are always
      # available.
      import svmutil
      svmutil.svm_save_model(root_path + '.svm', classifier)
    # Restore the value of the "classifier" and "pool" attributes.
    self.classifier = classifier
    self.pool = pool

  @staticmethod
  def Load(root_path):
    """Load the experiment from disk."""
    experiment = util.Load(root_path)
    if experiment.classifier != None:
      # Use delayed import of LIBSVM library, so non-SVM methods are always
      # available.
      import svmutil
      experiment.classifier = svmutil.svm_load_model(root_path + '.svm')
    return experiment

__POOL = None
__MODEL_CLASS = viz2.Model
__EXP = None
__VERBOSE = False

def UseCluster(config_file):
  """Use a cluster of worker nodes for any following experiment commands.
  config_file -- (str) path to the cluster configuration file
  """
  global __POOL
  from glimpse.pools.cluster import ClusterConfig, ClusterPool
  config = ClusterConfig(config_file)
  del __POOL
  __POOL = ClusterPool(config)

def SetModelClass(model_class):
  """Set the model type.
  model_class -- for example, use glimpse.models.viz2.model.Model
  """
  global __MODEL_CLASS
  __MODEL_CLASS = model_class

def MakeParams():
  """Create a default set of parameters for the current model type."""
  global __MODEL_CLASS
  return __MODEL_CLASS.Params()

def MakeModel(params = None):
  """Create the default model."""
  global __MODEL_CLASS
  if params == None:
    params = MakeParams()
  return __MODEL_CLASS(backends.MakeBackend(), params)

def GetExperiment():
  """Get the current experiment object."""
  global __EXP
  if __EXP == None:
    SetExperiment()
  return __EXP

def SetExperiment(model = None, layer = None, ordered = True):
  """Create a new experiment.
  model -- the Glimpse model to use for processing images
  layer -- (LayerSpec or str) the layer activity to use for features vectors
  ordered -- (bool) whether the order of SVM feature vectors must be preserved
             within each class
  """
  global __EXP, __POOL
  if __POOL == None:
    __POOL = pools.MakePool()
  if model == None:
    model = MakeModel()
  if layer == None:
    layer = model.Layer.IT
  elif isinstance(layer, str):
    layer = model.Layer.FromName(layer)
  __EXP = Experiment(model, layer, pool = __POOL, ordered = ordered)

def ImprintS2Prototypes(num_prototypes):
  """Imprint a set of S2 prototypes from a set of training images.
  num_prototypes -- (int) the number of C1 patches to sample
  """
  if __VERBOSE:
    print "Imprinting %d prototypes" % num_prototypes
  result = GetExperiment().ImprintS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def MakeRandomPrototypes(num_prototypes):
  """Create a set of S2 prototypes with uniformly random entries.
  num_prototypes -- (int) the number of S2 prototype arrays to create
  """
  if __VERBOSE:
    print "Making %d random prototypes" % num_prototypes
  result = GetExperiment().MakeRandomPrototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def SetCorpus(corpus_dir, classes = None):
  """Read images from the corpus directory, and choose training and testing
  subsets automatically. Use this instead of SetTrainTestSplit().
  corpus_dir -- (str) path to corpus directory
  classes -- (list) set of class names. Use this to ensure a given order to
             the SVM classes. When applying a binary SVM, the first class is
             treated as positive and the second class is treated as negative.
  """
  return GetExperiment().SetCorpus(corpus_dir, classes)

def SetTrainTestSplit(train_dir, test_dir, classes = None):
  """Read images from the corpus directories, setting the training and testing
  subsets manually. Use this instead of SetCorpus()."""
  return GetExperiment().SetTrainTestSplit(train_dir, test_dir, classes)

def TrainSvm():
  """Train an SVM classifier from the set of training images."""
  global __VERBOSE
  if __VERBOSE:
    print "Train SVM on %d images" % sum(map(len, GetExperiment().train_images))
  result = GetExperiment().TrainSvm()
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().svm_train_time
  return result

def TestSvm():
  """Apply the classifier to a set of test images.
  RETURN (float) accuracy
  """
  if __VERBOSE:
    print "Testing SVM on %d images" % \
        sum(map(len, GetExperiment().test_images))
  result = GetExperiment().TestSvm()
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().svm_test_time
  return result

def StoreExperiment(root_path):
  """Save the experiment to disk."""
  return GetExperiment().Store(root_path)

def LoadExperiment(root_path):
  """Load the experiment from disk."""
  global __EXP
  __EXP = Experiment.Load(root_path)
  return __EXP

def Verbose(flag):
  """Set (or unset) verbose logging."""
  global __VERBOSE
  __VERBOSE = flag
