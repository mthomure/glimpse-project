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
RunSvm()
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
from glimpse.backends import InsufficientSizeException
from glimpse.models import viz2
from glimpse import pools
from glimpse import util
from glimpse.util.grandom import HistogramSampler
from glimpse.util.svm import SpheringFeatureScaler, PrepareLibSvmInput, \
    SvmForSplit, SvmCrossValidate
from glimpse.models.misc import InputSourceLoadException
import itertools
import logging
import numpy as np
import operator
import os
import sys
import time

__all__ = ( 'SetPool', 'UseCluster', 'SetModelClass', 'SetParams', 'GetParams',
    'MakeParams', 'MakeModel', 'GetExperiment', 'SetExperiment',
    'ImprintS2Prototypes', 'MakeUniformRandomS2Prototypes',
    'MakeShuffledRandomS2Prototypes', 'MakeHistogramRandomS2Prototypes',
    'MakeNormalRandomS2Prototypes', 'SetS2Prototypes', 'SetCorpus',
    'SetTrainTestSplit', 'SetTrainTestSplitFromDirs', 'ComputeFeatures',
    'RunSvm', 'LoadExperiment', 'StoreExperiment', 'Verbose')

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
    self.train_features = None  # (list of 2D array) indexed by class, image,
                                # and then feature offset
    self.test_features = None  # (list of 2D array) indexed as in train_features
    self.train_results = None
    self.test_results = None
    self.cross_validated = None  # (bool) indicates whether cross-validation was
                                 # used to compute test accuracy.
    self.prototype_construction_time = None
    self.svm_train_time = None
    self.svm_test_time = None
    self.debug = False

  @property
  def features(self):
    """The full set of features for each class, without training/testing splits.
    RETURN (list of 2D float ndarray) indexed by class, image, and then feature
    offset.
    """
    if self.train_features == None:
      return None
    # Reorder instances from (set, class) indexing, to (class, set) indexing.
    features = zip(self.train_features, self.test_features)
    # Concatenate instances for each class (across sets)
    features = map(np.vstack, features)
    return features

  @property
  def images(self):
    """The full set of images, without training/testing splits.
    RETURN (list of string lists) indexed by class, and then image.
    """
    if self.train_images == None:
      return None
    # Combine images by class, and concatenate lists.
    return map(util.UngroupLists, zip(self.train_images, self.test_images))

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
    if self.train_results == None:
      values['train_accuracy'] = None
    else:
      values['train_accuracy'] = self.train_results['accuracy']
    values['test_accuracy'] = self.test_results['accuracy']
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
      sys.exit("Please specify the training corpus before imprinting "
          "prototypes.")
    start_time = time.time()
    image_files = util.UngroupLists(self.train_images)
    # Represent each image file as an empty model state.
    input_states = map(self.model.MakeStateFromFilename, image_files)
    try:
      prototypes, locations = self.model.ImprintS2Prototypes(num_prototypes,
          input_states, normalize = True, pool = self.pool)
    except InputSourceLoadException, e:
      logging.error("Failed to process image (%s): image read error" % \
          e.source.image_path)
      sys.exit(-1)
    except InsufficientSizeException, e:
      logging.error("Failed to process image (%s): image too small" % \
          e.source.image_path)
      sys.exit(-1)
    # Store new prototypes in model.
    self.prototype_source = 'imprinted'
    if self.debug:
      # Convert input source index to corresponding image path.
      locations = [ (image_files[l[0]],) + l[1:] for l in locations ]
      self.debug_prototype_locations = locations
    self.model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time

  def MakeUniformRandomS2Prototypes(self, num_prototypes):
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
    self.prototype_source = 'uniform'

  def MakeShuffledRandomS2Prototypes(self, num_prototypes):
    """Create a set of S2 prototypes by imprinting, and then shuffling the order
    of entries within each prototype.
    num_prototypes -- (int) the number of S2 prototype arrays to create
    """
    start_time = time.time()
    if self.model.s2_kernels == None:
      self.ImprintS2Prototypes(num_prototypes)
    for k in self.model.s2_kernels:
      np.random.shuffle(k.flat)
    self.prototype_construction_time = time.time() - start_time
    self.prototype_source = 'shuffle'

  def MakeHistogramRandomS2Prototypes(self, num_prototypes):
    """Create a set of S2 prototypes by drawing elements from a distribution,
    which is estimated from a set of imprinted prototypes. Each entry is drawn
    independently of the others.
    num_prototypes -- (int) the number of S2 prototype arrays to create
    """
    start_time = time.time()
    # Get ~100k C1 samples, which is approximately 255 prototypes.
    self.ImprintS2Prototypes(num_prototypes = 255)
    hist = HistogramSampler(self.model.s2_kernels.flat)
    size = (num_prototypes,) + self.model.s2_kernel_shape
    prototypes = hist.Sample(size).astype(
        util.ACTIVATION_DTYPE)
    for p in prototypes:
      p /= np.linalg.norm(p)
    self.model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time
    self.prototype_source = 'histogram'

  def MakeNormalRandomS2Prototypes(self, num_prototypes):
    """Create a set of S2 prototypes by drawing elements from the normal
    distribution, whose parameters are estimated from a set of imprinted
    prototypes. Each entry is drawn independently of the others.
    num_prototypes -- (int) the number of S2 prototype arrays to create
    """
    start_time = time.time()
    # Get ~100k C1 samples, which is approximately 255 prototypes.
    self.ImprintS2Prototypes(num_prototypes = 255)
    mean, std = self.model.s2_kernels.mean(), self.model.s2_kernels.std()
    size = (num_prototypes,) + self.model.s2_kernel_shape
    prototypes = np.random.normal(mean, std, size = size).astype(
        util.ACTIVATION_DTYPE)
    for p in prototypes:
      p /= np.linalg.norm(p)
    self.model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time
    self.prototype_source = 'normal'

  def SetS2Prototypes(self, prototypes):
    """Set the S2 prototypes from an array.
    prototypes -- (ndarray) the set of prototypes
    """
    self.prototype_source = 'manual'
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
    try:
      output_states = self.pool.map(builder, input_states)
    except InputSourceLoadException, e:
      logging.error("Failed to read image from disk: %s" % e.source.image_path)
      sys.exit(-1)
    except InsufficientSizeException, e:
      logging.error("Failed to process image (%s): image too small" % \
          e.source.image_path)
      sys.exit(-1)
    if self.debug:
      self.debug_output_states = output_states
    # Look up the activity values for the output layer, and convert them all to
    # a single vector.
    return [ util.ArrayListToVector(state[self.layer.id])
        for state in output_states ]

  def _ReadCorpusDir(self, corpus_dir, classes = None):
    if classes == None:
      classes = os.listdir(corpus_dir)
    try:
      def image_filter(img):
        # Ignore "hidden" files in corpus directory.
        return not img.startswith('.')
      def read_class(cls):
        class_dir = os.path.join(corpus_dir, cls)
        return [ os.path.join(class_dir, img) for img in os.listdir(class_dir)
            if image_filter(img) ]
      return map(read_class, classes)
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
    train_dir -- (str) path to directory of training images
    test_dir -- (str) path to directory of testing images
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
    train_images -- (list of str list) paths for each training image, with one
                    sub-list per class
    test_images -- (list of str list) paths for each training image, with one
                   sub-list per class
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
    train_images = util.UngroupLists(self.train_images)
    test_images = util.UngroupLists(self.test_images)
    images = train_images + test_images
    # Compute features for all images.
    input_states = map(self.model.MakeStateFromFilename, images)
    start_time = time.time()
    features = self.ComputeFeaturesFromInputStates(input_states)
    self.compute_feature_time = time.time() - start_time
    # Split results by training/testing set
    train_features, test_features = util.SplitList(features, [train_size])
    # Split training set by class
    train_features = util.SplitList(train_features, train_sizes)
    # Split testing set by class
    test_features = util.SplitList(test_features, test_sizes)
    # Store features as list of 2D arrays
    self.train_features = [ np.array(f, util.ACTIVATION_DTYPE)
        for f in train_features ]
    self.test_features = [ np.array(f, util.ACTIVATION_DTYPE)
        for f in test_features ]

  def RunSvm(self, cross_validate = False):
    """Train and test an SVM classifier from the set of training images.
    cross_validate -- (bool) if true, perform 10x10-way cross-validation.
                      Otherwise, compute accuracy for existing training/testing
                      split.
    RETURN (float tuple) training and testing accuracies (training accuracy is
    None when cross-validating.)
    """
    if self.train_features == None:
      self.ComputeFeatures()
    start_time = time.time()
    if cross_validate:
      test_accuracy = SvmCrossValidate(self.features, num_repetitions = 10,
          num_splits = 10, scaler = self.scaler)
      train_accuracy = None
      self.train_results = None
      self.test_results = dict(accuracy = test_accuracy)
    else:
      self.classifier, self.train_results, self.test_results = \
          SvmForSplit(self.train_features, self.test_features,
              scaler = self.scaler)
      train_accuracy = self.train_results['accuracy']
    self.cross_validated = cross_validate
    self.svm_time = time.time() - start_time
    return train_accuracy, self.test_results['accuracy']

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
    if self.debug and hasattr(pool, 'cluster_stats'):
      # This is a hackish way to record basic cluster usage information.
      util.Store(pool.cluster_stats, root_path + '.cluster-stats')
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
__MODEL_CLASS = None
__PARAMS = None
__LAYER = None
__EXP = None
__VERBOSE = False

def SetPool(pool):
  """Set the worker pool used for this experiment."""
  global __POOL
  logging.info("Using pool type: %s" % type(pool).__name__)
  __POOL = pool

def MakeClusterPool(config_file = None, chunksize = None):
  from glimpse.pools.gearman_cluster import ClusterConfig, ClusterPool
  if config_file == None:
    if 'GLIMPSE_CLUSTER_CONFIG' not in os.environ:
      raise ValueError("Please specify a cluster configuration file.")
    config_file = os.environ['GLIMPSE_CLUSTER_CONFIG']
  config = ClusterConfig(config_file)
  return ClusterPool(config, chunksize = chunksize)

def UseCluster(config_file = None, chunksize = None):
  """Use a cluster of worker nodes for any following experiment commands.
  config_file -- (str) path to the cluster configuration file
  """
  SetPool(MakeClusterPool(config_file, chunksize))

def SetModelClass(model_class = None):
  """Set the model type.
  model_class -- for example, use glimpse.models.viz2.model.Model
  """
  global __MODEL_CLASS
  if model_class == None:
    model_class = viz2.Model  # the default GLIMPSE model
  logging.info("Using model type: %s" % model_class.__name__)
  __MODEL_CLASS = model_class
  return __MODEL_CLASS

def GetModelClass():
  global __MODEL_CLASS
  if __MODEL_CLASS == None:
    SetModelClass()
  return __MODEL_CLASS

def SetParams(params = None):
  global __PARAMS
  if params == None:
    params = GetModelClass().Params()
  __PARAMS = params
  return __PARAMS

def GetParams():
  global __PARAMS
  if __PARAMS == None:
    SetParams()
  return __PARAMS

def SetLayer(layer = None):
  global __LAYER, __MODEL_CLASS
  if layer == None:
    layer = __MODEL_CLASS.Layer.IT
  elif isinstance(layer, str):
    layer = model.Layer.FromName(layer)
  __LAYER = layer
  return __LAYER

def GetLayer():
  global __LAYER
  if __LAYER == None:
    SetLayer()
  return __LAYER

def MakeModel(params = None):
  """Create the default model."""
  global __MODEL_CLASS
  if params == None:
    params = GetParams()
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
    layer = GetLayer()
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

def MakeUniformRandomS2Prototypes(num_prototypes):
  """Create a set of S2 prototypes with uniformly random entries.
  num_prototypes -- (int) the number of S2 prototype arrays to create
  """
  if __VERBOSE:
    print "Making %d uniform random prototypes" % num_prototypes
  result = GetExperiment().MakeUniformRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def MakeShuffledRandomS2Prototypes(num_prototypes):
  """Create a set of S2 prototypes by imprinting, and then shuffling the order
  of entries within each prototype.
  num_prototypes -- (int) the number of S2 prototype arrays to create
  """
  if __VERBOSE:
    print "Making %d shuffled random prototypes" % num_prototypes
  result = GetExperiment().MakeShuffledRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def MakeHistogramRandomS2Prototypes(num_prototypes):
  """Create a set of S2 prototypes by drawing elements from a distribution,
  which is estimated from a set of imprinted prototypes. Each entry is drawn
  independently of the others.
  num_prototypes -- (int) the number of S2 prototype arrays to create
  """
  if __VERBOSE:
    print "Making %d histogram random prototypes" % num_prototypes
  result = GetExperiment().MakeHistogramRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def MakeNormalRandomS2Prototypes(num_prototypes):
  """Create a set of S2 prototypes by drawing elements from the normal
  distribution, whose parameters are estimated from a set of imprinted
  prototypes. Each entry is drawn independently of the others.
  num_prototypes -- (int) the number of S2 prototype arrays to create
  """
  if __VERBOSE:
    print "Making %d normal random prototypes" % num_prototypes
  result = GetExperiment().MakeNormalRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def SetS2Prototypes(prototypes):
  """Set the S2 prototypes from an array or a file.
  prototypes -- (ndarray) the set of prototypes, or (str) a path to a file
                containing the prototypes
  """
  if isinstance(prototypes, basestring):
    prototypes = util.Load(prototypes)
  elif not isinstance(prototypes, np.ndarray):
    raise ValueError("Please specify an array of prototypes, or the path to a "
        "file.")
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
  """Compute SVM feature vectors for all images. Generally, you do not need to
  call this method yourself, as it will be called automatically by RunSvm()."""
  GetExperiment().ComputeFeatures()

def RunSvm(cross_validate = False):
  """Train and test an SVM classifier from the set of images in the corpus.
  cross_validate -- (bool) if true, perform 10x10-way cross-validation.
                    Otherwise, compute accuracy for existing training/testing
                    split.
  RETURN (float tuple) training and testing accuracies
  """
  global __VERBOSE
  e = GetExperiment()
  if cross_validate:
    if __VERBOSE:
      print "Computing cross-validated SVM performance on %d images" % \
          sum(map(len, e.images))
  else:
    if __VERBOSE:
      print "Train SVM on %d images" % sum(map(len, e.train_images))
      print "  and testing on %d images" % sum(map(len, e.test_images))
  train_accuracy, test_accuracy = e.RunSvm(cross_validate)
  if __VERBOSE:
    print "  done: %s s" % e.svm_time
    print "Time to compute feature vectors: %s s" % \
        e.compute_feature_time
  return train_accuracy, test_accuracy

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

def CLIGetModel(model_name):
  models = __import__("glimpse.models.%s" % model_name, globals(), locals(),
      ['Model'], 0)
  try:
    return getattr(models, 'Model')
  except AttributeError:
    raise util.UsageException("Unknown model (-m): %s" % model_name)

def CLIInit(pool_type = None, cluster_config = None, model_name = None,
    params = None, edit_params = False, layer = None, debug = False,
    verbose = 0, **opts):
  if verbose > 0:
    Verbose(True)
    if verbose > 1:
      logging.getLogger().setLevel(logging.INFO)
  # Make the worker pool
  if pool_type != None:
    pool_type = pool_type.lower()
    if pool_type in ('c', 'cluster'):
      pool = MakeClusterPool(cluster_config)
    elif pool_type in ('m', 'multicore'):
      pool = pools.MulticorePool()
    elif pool_type in ('s', 'singlecore'):
      pool = pools.SinglecorePool()
    else:
      raise util.UsageException("Unknown pool type: %s" % pool_type)
    SetPool(pool)
  if model_name != None:
    SetModelClass(CLIGetModel(model_name))
  SetParams(params)
  SetLayer(layer)
  if edit_params:
    GetParams().configure_traits()
  GetExperiment().debug = debug

def CLIFormatResults(svm_decision_values = False, svm_predicted_labels = False,
    **opts):
  e = GetExperiment()
  if e.train_results != None:
    print "Train Accuracy: %.3f" % e.train_results['accuracy']
  if e.test_results != None:
    print "Test Accuracy: %.3f" % e.test_results['accuracy']
    test_images = e.test_images
    test_results = e.test_results
    if svm_decision_values:
      if 'decision_values' not in test_results:
        logging.warn("Decision values are unavailable.")
      decision_values = test_results['decision_values']
      print "Decision Values:"
      for cls in range(len(test_images)):
        print "\n".join("%s %s" % _
            for _ in zip(test_images[cls], decision_values[cls]))
    if svm_predicted_labels:
      if 'predicted_labels' not in test_results:
        logging.warn("Decision values are unavailable.")
      predicted_labels = test_results['predicted_labels']
      print "Predicted Labels:"
      for cls in range(len(test_images)):
        print "\n".join("%s %s" % _
            for _ in zip(test_images[cls], predicted_labels[cls]))
  else:
    print "No results available."

def CLIRun(prototypes = None, prototype_algorithm = None, num_prototypes = 10,
    corpus = None, svm = False, compute_features = False, result_path = None,
    cross_validate = False, verbose = 0, **opts):
  if corpus != None:
    SetCorpus(corpus)
  num_prototypes = int(num_prototypes)
  if prototypes != None:
    SetS2Prototypes(prototypes)
  if prototype_algorithm != None:
    prototype_algorithm = prototype_algorithm.lower()
    if prototype_algorithm == 'imprint':
      ImprintS2Prototypes(num_prototypes)
    elif prototype_algorithm == 'uniform':
      MakeUniformRandomS2Prototypes(num_prototypes)
    elif prototype_algorithm == 'shuffle':
      MakeShuffledRandomS2Prototypes(num_prototypes)
    elif prototype_algorithm == 'histogram':
      MakeHistogramRandomS2Prototypes(num_prototypes)
    elif prototype_algorithm == 'normal':
      MakeNormalRandomS2Prototypes(num_prototypes)
    else:
      raise util.UsageException("Invalid prototype algorithm "
          "(%s), expected 'imprint' or 'random'." % prototype_algorithm)
  if compute_features:
    ComputeFeatures()
  if svm:
    RunSvm(cross_validate)
    if verbose > 0:
      CLIFormatResults(**opts)
  if result_path != None:
    StoreExperiment(result_path)

def CLI(**opts):
  """Entry point for command-line interface handling."""
  CLIInit(**opts)
  CLIRun(**opts)

def main():
  default_model = "viz2"
  try:
    opts = dict()
    opts['verbose'] = 0
    result_path = None
    verbose = 0
    cli_opts, cli_args = util.GetOptions('c:C:del:m:n:o:p:P:r:st:vx',
        ['corpus=', 'cluster-config=', 'compute-features', 'debug',
        'edit-options', 'layer=', 'model=', 'num-prototypes=', 'options=',
        'prototype-algorithm=', 'prototypes=', 'results=', 'svm',
        'svm-decision-values', 'svm-predicted-labels', 'pool-type=', 'verbose',
        'cross-validate'])
    for opt, arg in cli_opts:
      if opt in ('-c', '--corpus'):
        opts['corpus'] = arg
      elif opt in ('-C', '--cluster-config'):
        # Use a cluster of worker nodes
        opts['cluster_config'] = arg
      elif opt in ('--compute-features'):
        opts['compute_features'] = True
      elif opt in ('-d', '--debug'):
        opts['debug'] = True
      elif opt in ('-e', '--edit-options'):
        opts['edit_params'] = True
      elif opt in ('-l', '--layer'):
        opts['layer'] = arg
      elif opt in ('-m', '--model'):
        # Set the model class
        if arg == 'default':
          arg = default_model
        opts['model_name'] = arg
      elif opt in ('-n', '--num-prototypes'):
        opts['num_prototypes'] = int(arg)
      elif opt in ('-o', '--options'):
        opts['params'] = util.Load(arg)
      elif opt in ('-p', '--prototype-algorithm'):
        opts['prototype_algorithm'] = arg.lower()
      elif opt in ('-P', '--prototypes'):
        opts['prototypes'] = util.Load(arg)
      elif opt in ('-r', '--results'):
        opts['result_path'] = arg
      elif opt in ('-s', '--svm'):
        opts['svm'] = True
      elif opt == '--svm-decision-values':
        opts['svm_decision_values'] = True
      elif opt == '--svm-predicted-labels':
        opts['svm_predicted_labels'] = True
      elif opt in ('-t', '--pool-type'):
        opts['pool_type'] = arg.lower()
      elif opt in ('-v', '--verbose'):
        opts['verbose'] += 1
      elif opt in ('-x', '--cross-validate'):
        opts['cross_validate'] = True
    CLI(**opts)
  except util.UsageException, e:
    util.Usage("[options]\n"
        "  -c, --corpus=DIR                Use corpus directory DIR\n"
        "  -C, --cluster-config=FILE       Read cluster configuration from "
        "FILE\n"
        "      --compute-features          Compute feature vectors (implied "
        "by -s)\n"
        "  -d, --debug                     Enable debugging\n"
        "  -e, --edit-options              Edit model options with a GUI\n"
        "  -l, --layer=LAYR                Compute feature vectors from LAYR "
        "activity\n"
        "  -m, --model=MODL                Use model named MODL\n"
        "  -n, --num-prototypes=NUM        Generate NUM S2 prototypes\n"
        "  -o, --options=FILE              Read model options from FILE\n"
        "  -p, --prototype-algorithm=ALG   Generate S2 prototypes according "
        "to algorithm\n"
        "                                  ALG (one of 'imprint', 'uniform', "
        "'shuffle',\n"
        "                                  'histogram', or 'normal')\n"
        "  -P, --prototypes=FILE           Read S2 prototypes from FILE "
        "(overrides -p)\n"
        "  -r, --results=FILE              Store results to FILE\n"
        "  -s, --svm                       Train and test an SVM classifier\n"
        "      --svm-decision-values       Print the pre-thresholded SVM "
        "decision values\n"
        "                                  for each test image\n"
        "      --svm-predicted-labels      Print the predicted labels for each "
        "test image\n"
        "  -t, --pool-type=TYPE            Set the worker pool type (one of "
        "'multicore',\n"
        "                                  'singlecore', or 'cluster')\n"
        "  -v, --verbose                   Enable verbose logging\n"
        "  -x, --cross-validate            Compute test accuracy via cross-"
        "validation\n"
        "                                  instead of fixed training/testing "
        "split",
        e
    )

if __name__ == '__main__':
  main()
