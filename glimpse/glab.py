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
from glimpse import pools
from glimpse import util
import itertools
import logging
import numpy as np
import operator
import os
import sys
import time

__all__ = ( 'SetPool', 'UseCluster', 'SetModelClass', 'MakeParams', 'MakeModel',
    'GetExperiment', 'SetExperiment', 'ImprintS2Prototypes',
    'MakeRandomS2Prototypes', 'SetS2Prototypes', 'SetCorpus',
    'SetTrainTestSplit', 'SetTrainTestSplitFromDirs', 'ComputeFeatures',
    'TrainSvm', 'TestSvm', 'LoadExperiment', 'StoreExperiment', 'Verbose')

def ChainLists(*iterables):
  """Concatenate several sequences to form a single list."""
  return list(itertools.chain(*iterables))

def ChainArrays(*arrays):
  """Concatenate several numpy arrays."""
  return np.vstack(arrays)

def SplitList(data, *sizes):
  """Break a list into sublists.
  data -- (list) input data
  sizes -- (int list) size of each chunk. if sum of sizes is less than entire
           size of input array, the remaining elements are returned as an extra
           sublist in the result.
  RETURN (list of lists) sublists of requested size
  """
  assert(all([ s >= 0 for s in sizes ]))
  if len(sizes) == 0:
    return data
  if sum(sizes) < len(data):
    sizes = list(sizes)
    sizes.append(len(data) - sum(sizes))
  out = list()
  last = 0
  for s in sizes:
    out.append(data[last : last+s])
    last += s
  return out

class RangeFeatureScaler(object):
  """Scales features to lie in a fixed interval."""

  def __init__(self, min = -1, max = 1):
    """Create new object.
    min -- (float) minimum value in output range
    max -- (float) maximum value in output range
    """
    self.omin, self.omax = min, max

  def Learn(self, features):
    """Determine the parameters required to scale each feature (independently)
    to the range [-1, 1].
    features -- (list of list)
    """
    self.imin, self.imax = np.min(features, 0), np.max(features, 0)

  def Apply(self, features):
    """Scale the features in-place. The range of output values will be
    (approximately) [-vmin, vmax], assuming the feature vectors passed here were
    drawn from the same distribution as those used to learn the scaling
    parameters.
    features -- (list of ndarray)
    RETURN (np.ndarray) new array of scaled feature values
    """
    features = np.array(features)  # copy feature values
    for f in features:
      f -= self.imin  # map to [0, imax - imin]
      f /= (self.imax - self.imin)  # map to [0, 1]
      f *= (self.omax - self.omin)  # map to [0, omax - omin]
      f += self.omin  # map to [omin, omax]
    return features

class SpheringFeatureScaler(object):
  """Scales features to have fixed mean and standard deviation."""

  def __init__(self, mean = 0, std = 1):
    """Create new object.
    mean -- (float) mean of output feature values
    std -- (float) standard deviation of feature values
    """
    self.omean, self.ostd = mean, std

  def Learn(self, features):
    """Determine the parameters required to scale each feature (independently)
    to the range [-1, 1].
    features -- (list of list)
    """
    self.imean, self.istd = np.mean(features, 0), np.std(features, 0)

  def Apply(self, features):
    """Scale the features. The range of output values will be (approximately)
    [-vmin, vmax], assuming the feature vectors passed here were drawn from the
    same distribution as those used to learn the scaling parameters.
    features -- (list of ndarray)
    RETURN (ndarray) new array with scaled features
    """
    features = np.array(features)  # copy feature values
    features -= self.imean  # map to mean zero
    features /= self.istd  # map to unit variance
    features *= self.ostd  # map to output standard deviation
    features += self.omean  # map to output mean
    return features

class Experiment(object):

  def __init__(self, model, layer, pool, scaler):
    """Create a new experiment.
    model -- the Glimpse model to use for processing images
    layer -- (LayerSpec) the layer activity to use for features vectors
    pool -- a serializable worker pool
    scaler -- feature scaling algorithm
    """
    # Default arguments should be chosen in SetExperiment()
    assert model != None
    assert layer != None
    assert pool != None
    assert scaler != None
    self.model = model
    self.pool = pool
    self.layer = layer
    self.scaler = scaler
    # Initialize attributes used by an experiment
    self.classes = []
    self.classifier = None
    self.corpus = None
    self.prototype_source = None
    self.train_images = None
    self.test_images = None
    self.train_test_split = None
    self.train_features = None
    self.test_features = None
    self.train_accuracy = None
    self.test_accuracy = None
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
  train_accuracy: %(train_accuracy)s
  test_accuracy: %(test_accuracy)s""" % values

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

  def MakeRandomS2Prototypes(self, num_prototypes):
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

  def SetS2Prototypes(self, prototypes):
    """Set the S2 prototypes from an array.
    prototypes -- (ndarray) the set of prototypes
    """
    self.model.s2_kernels = prototypes

  def ComputeFeaturesFromInputStates(self, input_states):
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
    output_states = self.pool.map(builder, input_states)
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

  def SetTrainTestSplitFromDirs(self, train_dir, test_dir, classes = None):
    """Read images from the corpus directories, setting the training and testing
    subsets manually. Use this instead of SetCorpus().
    train_dir -- (str)
    test_dir -- (str)
    classes -- (list) class names
    """
    if classes == None:
      classes = os.listdir(train_dir)
    train_images = self._ReadCorpusDir(train_dir, classes)
    test_images = self._ReadCorpusDir(test_dir, classes)
    self.SetTrainTestSplit(train_images, test_images, classes)
    self.corpus = (train_dir, test_dir)

  def SetTrainTestSplit(self, train_images, test_images, classes):
    """Manually specify the training and testing images.
    train_images -- (str)
    test_images -- (str)
    classes -- (list) class names
    """
    self.classes = classes
    self.train_test_split = 'manual'
    self.train_images = train_images
    self.test_images = test_images

  def ComputeFeatures(self):
    """Compute SVM feature vectors for all images."""
    if self.train_images == None or self.test_images == None:
      sys.exit("Please specify the corpus.")
    train_sizes = map(len, self.train_images)
    train_size = sum(train_sizes)
    test_sizes = map(len, self.test_images)
    train_images = ChainLists(*self.train_images)
    test_images = ChainLists(*self.test_images)
    images = train_images + test_images
    # Compute features for all images.
    input_states = map(self.model.MakeStateFromFilename, images)
    start_time = time.time()
    features = self.ComputeFeaturesFromInputStates(input_states)
    self.compute_feature_time = time.time() - start_time
    train_features, test_features = SplitList(features, train_size)
    self.train_features = np.array(SplitList(train_features, *train_sizes),
        util.ACTIVATION_DTYPE)
    self.test_features = np.array(SplitList(test_features, *test_sizes),
        util.ACTIVATION_DTYPE)

  def PrepareLibSvmInput(self, features_per_class):
    """Format feature vectors for use by LIBSVM.
    features_per_class -- (list of ndarray) per-class feature vectors
    RETURN (tuple) (list) labels, and (list) all feature vectors
    """
    num_classes = len(features_per_class)
    if num_classes == 2:
      labels = [1, -1]
    else:
      labels = range(1, num_classes + 1)
    class_sizes = map(len, features_per_class)
    labels_per_class = [ [label] * size
        for label, size in zip(labels, class_sizes) ]
    # Convert feature vectors from np.array to list objects
    features_per_class = [ map(list, features)
        for features in features_per_class ]
    return ChainLists(*labels_per_class), ChainLists(*features_per_class)

  def TrainSvm(self):
    """Train an SVM classifier from the set of training images."""
    if self.train_features == None:
      self.ComputeFeatures()
    start_time = time.time()
    # learn from single list of feature vectors
    self.scaler.Learn(ChainArrays(*self.train_features))
    # sphere features
    features = map(self.scaler.Apply, self.train_features)
    svm_labels, svm_features = self.PrepareLibSvmInput(features)
    options = '-q'  # don't write to stdout
    # Use delayed import of LIBSVM library, so non-SVM methods are always
    # available.
    import svmutil
    self.classifier = svmutil.svm_train(svm_labels, svm_features, options)
    options = ''  # can't disable writing to stdout
    print "LIBSVM-INTERNAL",  # mark output from LIBSVM
    predicted_labels, acc, decision_values = svmutil.svm_predict(svm_labels,
        svm_features, self.classifier, options)
    self.train_accuracy = float(acc[0]) / 100.
    self.svm_train_time = time.time() - start_time
    return self.train_accuracy

  def TestSvm(self):
    """Apply the classifier to a set of test images.
    RETURN (float) accuracy
    """
    if self.classifier == None:
      sys.exit("Please train the classifier before calling TestSvm().")
    if self.test_features == None:
      sys.exit("Internal error: missing SVM features for test images.")
    start_time = time.time()
    # sphere features
    features = map(self.scaler.Apply, self.test_features)
    svm_labels, svm_features = self.PrepareLibSvmInput(features)
    options = ''  # can't disable writing to stdout
    # Use delayed import of LIBSVM library, so non-SVM methods are always
    # available.
    import svmutil
    print "LIBSVM-INTERNAL",  # mark output from LIBSVM
    predicted_labels, acc, decision_values = svmutil.svm_predict(svm_labels,
        svm_features, self.classifier, options)
    # Ignore mean-squared error and correlation coefficient
    self.test_accuracy = float(acc[0]) / 100.
    self.svm_test_time = time.time() - start_time
    return self.test_accuracy

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

def SetPool(pool):
  """Set the worker pool used for this experiment."""
  global __POOL
  logging.info("Using pool type: %s" % type(pool).__name__)
  __POOL = pool

def UseCluster(config_file = None, chunksize = None):
  """Use a cluster of worker nodes for any following experiment commands.
  config_file -- (str) path to the cluster configuration file
  """
  from glimpse.pools.cluster import ClusterConfig, ClusterPool
  if config_file == None:
    if 'GLIMPSE_CLUSTER_CONFIG' not in os.environ:
      raise ValueError("Please specify a cluster configuration file.")
    config_file = os.environ['GLIMPSE_CLUSTER_CONFIG']
  config = ClusterConfig(config_file)
  SetPool(ClusterPool(config, chunksize = chunksize))

def SetModelClass(model_class):
  """Set the model type.
  model_class -- for example, use glimpse.models.viz2.model.Model
  """
  global __MODEL_CLASS
  logging.info("Using model type: %s" % model_class.__name__)
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

def SetExperiment(model = None, layer = None, scaler = None):
  """Create a new experiment.
  model -- the Glimpse model to use for processing images
  layer -- (LayerSpec or str) the layer activity to use for features vectors
  scaler -- feature scaling algorithm
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
  if scaler == None:
    scaler = SpheringFeatureScaler()
  __EXP = Experiment(model, layer, pool = __POOL, scaler = scaler)

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

def MakeRandomS2Prototypes(num_prototypes):
  """Create a set of S2 prototypes with uniformly random entries.
  num_prototypes -- (int) the number of S2 prototype arrays to create
  """
  if __VERBOSE:
    print "Making %d random prototypes" % num_prototypes
  result = GetExperiment().MakeRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def SetS2Prototypes(prototypes):
  """Set the S2 prototypes from an array or a file.
  prototypes -- (ndarray) the set of prototypes, or (str) a path to a file containing the prototypes
  """
  if isinstance(prototypes, basestring):
    prototypes = util.Load(prototypes)
  elif not isinstance(prototypes, np.ndarray):
    raise ValueError("Please specify an array of prototypes, or the path to a file.")
  GetExperiment().SetS2Prototypes(prototypes)

def SetCorpus(corpus_dir, classes = None):
  """Read images from the corpus directory, and choose training and testing
  subsets automatically. Use this instead of SetTrainTestSplit().
  corpus_dir -- (str) path to corpus directory
  classes -- (list) set of class names. Use this to ensure a given order to
             the SVM classes. When applying a binary SVM, the first class is
             treated as positive and the second class is treated as negative.
  """
  return GetExperiment().SetCorpus(corpus_dir, classes)

def SetTrainTestSplitFromDirs(train_dir, test_dir, classes = None):
  """Read images from the corpus directories, setting the training and testing
  subsets manually. Use this instead of SetCorpus().
  """
  return GetExperiment().SetTrainTestSplit(train_dir, test_dir, classes)

def SetTrainTestSplit(train_images, test_images, classes):
  """Manually specify the training and testing images."""
  return GetExperiment().SetTrainTestSplit(train_images, test_images, classes)

def ComputeFeatures():
  """Compute SVM feature vectors for all images. Generally, you do not need to call this method yourself, as it will be called automatically by TrainSvm()."""
  GetExperiment().ComputeFeatures()

def TrainSvm():
  """Train an SVM classifier from the set of training images."""
  global __VERBOSE
  if __VERBOSE:
    print "Train SVM on %d images" % sum(map(len, GetExperiment().train_images))
  result = GetExperiment().TrainSvm()
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().svm_train_time
    print "Time to compute train/test feature vectors: %s s" % \
        GetExperiment().compute_feature_time
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

#### CLI Interface ####

def __CLIGetModel(model_name):
  models = __import__("glimpse.models.%s" % model_name, globals(), locals(),
      ['Model'], 0)
  try:
    return getattr(models, 'Model')
  except AttributeError:
    raise util.UsageException("Unknown model (-m): %s" % model_name)

def __CLIMakeClusterPool(config_file = None):
  from glimpse.pools.cluster import ClusterConfig, ClusterPool
  if config_file == None:
    if 'GLIMPSE_CLUSTER_CONFIG' not in os.environ:
      raise util.UsageException("Please specify a cluster configuration file.")
    config_file = os.environ['GLIMPSE_CLUSTER_CONFIG']
  return ClusterPool(ClusterConfig(config_file))

def __CLIInit(pool_type = None, cluster_config = None, model_name = None, params = None,
    edit_params = False, layer = None, **opts):
  # Make the worker pool
  if pool_type != None:
    pool_type = pool_type.lower()
    if pool_type in ('c', 'cluster'):
      pool = __CLIMakeClusterPool(cluster_config)
    elif pool_type in ('m', 'multicore'):
      pool = pools.MulticorePool()
    elif pool_type in ('s', 'singlecore'):
      pool = pools.SinglecorePool()
    else:
      raise util.UsageException("Unknown pool type: %s" % pool_type)
    SetPool(pool)
  if model_name != None:
    SetModelClass(__CLIGetModel(model_name))
  model = None
  if edit_params:
    if params == None:
      params = MakeParams()
      params.configure_traits()
      model = MakeModel(params)
  elif params != None:
    model = MakeModel(params)
  if model != None or layer != None:
    SetExperiment(model = model, layer = layer)

def __CLIRun(prototypes = None, prototype_algorithm = None, num_prototypes = 10, corpus = None,
    svm = False, **opts):
  if corpus != None:
    SetCorpus(corpus)
  num_prototypes = int(num_prototypes)
  if prototypes != None:
    SetS2Prototypes(prototypes)
  elif prototype_algorithm != None:
    prototype_algorithm = prototype_algorithm.lower()
    if prototype_algorithm == 'imprint':
      ImprintS2Prototypes(num_prototypes)
    elif prototype_algorithm == 'random':
      MakeRandomS2Prototypes(num_prototypes)
    else:
      raise util.UsageException("Invalid prototype algorithm (%s), expected 'imprint' "
                              "or 'random'." % prototype_algorithm)
  if svm:
    train_accuracy = TrainSvm()
    test_accuracy = TestSvm()
    print "Train Accuracy: %.3f" % train_accuracy
    print "Test Accuracy: %.3f" % test_accuracy

def main():
  default_model = "viz2"
  try:
    opts = dict()
    result_path = None
    verbose = 0
    cli_opts, cli_args = util.GetOptions('c:C:el:m:n:o:p:P:r:st:v', ['corpus=', 'cluster_config=',
        'edit_options', 'layer=', 'model=', 'num_prototypes=', 'options=', 'prototype_algorithm=',
        'prototypes=', 'results=', 'svm', 'pool_type=', 'verbose'])
    for opt, arg in cli_opts:
      if opt in ('-c', '--corpus'):
        opts['corpus'] = arg
      elif opt in ('-C', '--cluster_config'):
        # Use a cluster of worker nodes
        opts['cluster_config'] = arg
      elif opt in ('-e', '--edit_options'):
        opts['edit_params'] = True
      elif opt in ('-l', '--layer'):
        opts['layer'] = arg
      elif opt in ('-m', '--model'):
        # Set the model class
        if arg == 'default':
          arg = default_model
        opts['model_name'] = arg
      elif opt in ('-n', '--num_prototypes'):
        opts['num_prototypes'] = int(arg)
      elif opt in ('-o', '--options'):
        opts['params'] = util.Load(arg)
      elif opt in ('-p', '--prototype_algorithm'):
        opts['prototype_algorithm'] = arg.lower()
      elif opt in ('-P', '--prototypes'):
        opts['prototypes'] = util.Load(arg)
      elif opt in ('-r', '--results'):
        result_path = arg
      elif opt in ('-s', '--svm'):
        opts['svm'] = True
      elif opt in ('-t', '--pool_type'):
        opts['pool_type'] = arg.lower()
      elif opt in ('-v', '--verbose'):
        verbose += 1
    if verbose > 0:
      Verbose(True)
      if verbose > 1:
        logging.getLogger().setLevel(logging.INFO)
    __CLIInit(**opts)
    __CLIRun(**opts)
    if result_path != None:
      StoreExperiment(result_path)
  except util.UsageException, e:
    util.Usage("[options]\n"
        "  -c, --corpus=DIR                Use corpus directory DIR\n"
        "  -C, --cluster_config=FILE       Read cluster configuration from FILE\n"
        "  -e, --edit_options              Edit model options with a GUI\n"
        "  -l, --layer=LAYR                Compute feature vectors from LAYR activity\n"
        "  -m, --model=MODL                Use model named MODL\n"
        "  -n, --num_prototypes=NUM        Generate NUM S2 prototypes\n"
        "  -o, --options=FILE              Read model options from FILE\n"
        "  -p, --prototype_algorithm=ALG   Generate S2 prototypes according to algorithm ALG\n"
        "  -P, --prototypes=FILE           Read S2 prototypes from FILE\n"
        "  -r, --results=FILE              Store results to FILE\n"
        "  -s, --svm                       Train and test an SVM classifier\n"
        "  -t, --pool_type=TYPE            Set the worker pool type\n"
        "  -v, --verbose                   Enable verbose logging",
        e
    )

if __name__ == '__main__':
  main()
