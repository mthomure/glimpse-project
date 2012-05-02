#!/usr/bin/python

"""This module provides a simplified interface for running Glimpse experiments.

The easiest way to use this module is via the top-level functions, such as
:func:`SetCorpus`, which provide a declarative interface similar to
Matlab(TM). Alternatively, an object-oriented interface is also available by
using the :class:`Experiment` class directly.

"""

from glimpse import backends
from glimpse.backends import InsufficientSizeException
from glimpse.models import viz2
from glimpse import pools
from glimpse import util
from glimpse.util import docstring
from glimpse.util.grandom import HistogramSampler
from glimpse.util import svm
from glimpse.models.misc import InputSourceLoadException
import itertools
import logging
import numpy as np
import operator
import os
import sys
import time

__all__ = ( 'SetPool', 'UseCluster', 'SetModelClass', 'SetParams', 'GetParams',
    'MakeModel', 'GetExperiment', 'SetExperiment', 'ImprintS2Prototypes',
    'MakeUniformRandomS2Prototypes', 'MakeShuffledRandomS2Prototypes',
    'MakeHistogramRandomS2Prototypes', 'MakeNormalRandomS2Prototypes',
    'SetS2Prototypes', 'SetCorpus', 'SetTrainTestSplit', 'SetLayer',
    'SetTrainTestSplitFromDirs', 'ComputeFeatures', 'CrossValidateSvm',
    'TrainSvm', 'TestSvm', 'RunSvm', 'LoadExperiment', 'StoreExperiment',
    'Verbose')

class Experiment(object):
  """Container for experimental results.

  For example, this object will contain the Glimpse model, the worker pool,
  the feature vectors, and the SVM results.

  """

  def __init__(self, model, layer, pool, scaler):
    """Create a new experiment.

    :type model: model object
    :param model: The Glimpse model to use for processing images.
    :param LayerSpec layer: The layer activity to use for features vectors.
    :type pool: pool object
    :param pool: A serializable worker pool.
    :param svm.Scaler scaler: Feature scaling algorithm.

    """
    # Default arguments should be chosen in SetExperiment()
    assert model != None
    assert layer != None
    assert pool != None
    assert scaler != None
    #: The Glimpse model used to compute feature vectors.
    self.model = model
    #: The worker pool used for parallelization.
    self.pool = pool
    #: The model layer from which feature vectors are extracted.
    self.layer = layer
    #: The algorithm used to scale feature vectors.
    self.scaler = scaler
    # Initialize attributes used by an experiment
    #: (list of str) Names of image classes.
    self.classes = []
    #: The built SVM classifier.
    self.classifier = None
    #: (str) Path to the corpus directory. May be empty if
    #: :meth:`SetCorpusSubdirs` was used.
    self.corpus = None
    #: (str) The method used to construct prototypes.
    self.prototype_source = None
    #: (list of list of str) The set of images used for training, indexed by
    #: class and then image offset.
    self.train_images = None
    #: (list of list of str) The set of images used for testing, indexed by
    #: class and then image offset.
    self.test_images = None
    #: (str) The method used to split the data into training and testing sets.
    self.train_test_split = None
    #: (list of 2D ndarray) Feature vectors for the training images, indexed by
    #: class, image, and then feature offset.
    self.train_features = None
    #: (int) If image resizing is to be used, this is the extent of the smaller
    #: edge.
    self.resize = None
    #: (list of 2D ndarray) Feature vectors for the test images, indexed by
    #: class, image, and then feature offset.
    self.test_features = None
    #: (dict) SVM evaluation results on the training data, in the format
    #: returned by :func:`Svm.Test <glimpse.util.svm.Svm.Test>`.
    self.train_results = None
    #: (dict) SVM evaluation results on the test data, in the format returned by
    #: :func:`Svm.Test <glimpse.util.svm.Svm.Test>`.
    self.test_results = None
    #: (bool) Flag indicating whether cross-validation was used to compute test
    #: accuracy.
    self.cross_validated = False
    #: (float) Time required to build S2 prototypes, in seconds.
    self.prototype_construction_time = None
    #: (float) Time required to train the SVM, in seconds.
    self.svm_train_time = None
    #: (float) Time required to test the SVM, in seconds.
    self.svm_test_time = None
    self.debug = False

  def GetFeatures(self):
    """The full set of features for each class, without training/testing splits.

    :rtype: list of 2D ndarray of float
    :returns: Copy of feature data, indexed by class, image, and then feature
       offset.

    """
    if self.train_features == None:
      return None
    # Reorder instances from (set, class) indexing, to (class, set) indexing.
    features = zip(self.train_features, self.test_features)
    # Concatenate instances for each class (across sets)
    features = map(np.vstack, features)
    return features

  def GetImages(self):
    """The full set of images, without training/testing splits.

    :rtype: list of list of str
    :returns: Copy of image paths, indexed by class, and then image.

    """
    if self.train_images == None:
      return None
    # Combine images by class, and concatenate lists.
    return map(util.UngroupLists, zip(self.train_images, self.test_images))

  @property
  def s2_prototypes(self):
    """The set of S2 prototypes."""
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
    if self.test_results == None:
      values['test_accuracy'] = None
    else:
      values['test_accuracy'] = self.test_results['accuracy']
    return """Experiment:
  corpus: %(corpus)s
  classes: %(classes)s
  train_test_split: %(train_test_split)s
  model: %(model)s
  layer: %(layer)s
  resize: %(resize)s
  prototype_source: %(prototype_source)s
  train_accuracy: %(train_accuracy)s
  test_accuracy: %(test_accuracy)s""" % values

  __repr__ = __str__

  def ImprintS2Prototypes(self, num_prototypes):
    """Imprint a set of S2 prototypes from a set of training images.

    :param int num_prototypes: The number of C1 patches to sample.

    """
    if self.train_images == None:
      sys.exit("Please specify the training corpus before imprinting "
          "prototypes.")
    start_time = time.time()
    image_files = util.UngroupLists(self.train_images)
    # Represent each image file as an empty model state.
    input_states = [ self.model.MakeStateFromFilename(f, resize = self.resize)
        for f in image_files ]
    try:
      prototypes, locations = self.model.ImprintS2Prototypes(num_prototypes,
          input_states, normalize = True, pool = self.pool)
    except InputSourceLoadException, e:
      if e.source != None:
        path = e.source.image_path
      else:
        path = '<unknown>'
      logging.error("Failed to process image (%s): image read error" % \
          path)
      sys.exit(-1)
    except InsufficientSizeException, e:
      if e.source != None:
        path = e.source.image_path
      else:
        path = '<unknown>'
      logging.error("Failed to process image (%s): image too small" % \
          path)
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

    :param int num_prototypes: The number of S2 prototype arrays to create.

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
    """Create a set of S2 prototypes.

    The set is chosen by imprinting, and then shuffling the order of entries
    within each prototype.

    :param int num_prototypes: The number of S2 prototype arrays to create.

    """
    start_time = time.time()
    if self.model.s2_kernels == None:
      self.ImprintS2Prototypes(num_prototypes)
    for k in self.model.s2_kernels:
      np.random.shuffle(k.flat)
    self.prototype_construction_time = time.time() - start_time
    self.prototype_source = 'shuffle'

  def MakeHistogramRandomS2Prototypes(self, num_prototypes):
    """Create a set of S2 prototypes.

    The set is created by drawing elements from a distribution, which is
    estimated from a set of imprinted prototypes. Each entry is drawn
    independently of the others.

    :param int num_prototypes: The number of S2 prototype arrays to create.

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
    """Create a set of S2 prototypes.

    The set is created by drawing elements from the normal distribution, whose
    parameters are estimated from a set of imprinted prototypes. Each entry is
    drawn independently of the others.

    :param int num_prototypes: The number of S2 prototype arrays to create.

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

    :param ndarray prototypes: The set of prototypes.

    """
    self.prototype_source = 'manual'
    self.model.s2_kernels = prototypes

  def ComputeFeaturesFromInputStates(self, input_states, block = True):
    """Return the activity of the model's output layer for a set of images.

    :type input_states: iterable of State
    :param input_states: Model states containing image data.
    :rtype: iterable
    :returns: A feature vector for each image.

    """
    L = self.model.LayerClass
    if self.layer in (L.S2, L.C2, L.IT) and self.model.s2_kernels == None:
      sys.exit("Please set the S2 prototypes before computing feature vectors "
          "for layer %s." % self.layer.name)
    builder = self.model.BuildLayerCallback(self.layer, save_all = False)
    # Compute model states containing desired features.
    output_states = self.pool.imap(builder, input_states)
    # Look up the activity values for the output layer, and convert them all to
    # a single vector.
    features = ( util.ArrayListToVector(state[self.layer.id])
        for state in output_states )
    if block:
      try:
        features = list(features)  # wait for results
      except InputSourceLoadException, e:
        logging.error("Failed to read image from disk: %s" % \
            e.source.image_path)
        sys.exit(-1)
      except InsufficientSizeException, e:
        logging.error("Failed to process image (%s): image too small" % \
            e.source.image_path)
        sys.exit(-1)
    return features

  def SetCorpus(self, corpus_dir, classes = None, balance = False):
    """Read images from the corpus directory.

    Training and testing subsets are chosen automatically.

    :param str corpus_dir: Path to corpus directory.
    :type classes: list of str
    :param classes: Set of class names. Use this to ensure a given order to the
       SVM classes. When applying a binary SVM, the first class is treated as
       positive and the second class is treated as negative.
    :param bool balance: Ensure an equal number of images from each class (by
       random selection).

    .. seealso::
       :func:`SetTrainTestSplit`

    """
    corpus_subdirs = [ os.path.join(corpus_dir, cls)
        for cls in os.listdir(corpus_dir) ]
    return self.SetCorpusSubdirs(corpus_subdirs, corpus_dir, classes, balance)

  def SetCorpusSubdirs(self, corpus_subdirs, corpus = None, classes = None,
      balance = False):
    """Read images from the corpus directory.

    Training and testing subsets are chosen automatically.

    :param corpus_subdirs: Paths to directories for each class.
    :type corpus_subdirs: list of str
    :param corpus: Path to main corpus directory.
    :type corpus: str, optional
    :type classes: list of str, optional
    :param classes: Set of class names. Use this to ensure a given order to the
       SVM classes. When applying a binary SVM, the first class is treated as
       positive and the second class is treated as negative.
    :type balance: bool, optional
    :param balance:  Ensure an equal number of images from each class (by random
       selection).

    .. seealso::
       :func:`SetTrainTestSplit`

    """
    if classes == None:
      classes = map(os.path.basename, corpus_subdirs)
    self.classes = classes
    self.corpus = corpus
    self.train_test_split = 'automatic'

    def image_filter(img):
      # Ignore "hidden" files in corpus directory.
      return not img.startswith('.')

    def read_subdir(path):
      return [ os.path.join(path, img)
          for img in os.listdir(path) if image_filter(img) ]

    try:
      images_per_class = map(read_subdir, corpus_subdirs)
    except OSError, e:
      sys.exit("Failed to read corpus directory: %s" % e)
    # Randomly reorder image lists.
    for images in images_per_class:
      np.random.shuffle(images)
    if balance:
      # Make sure each class has the same number of images
      num_images = map(len, images_per_class)
      size = min(num_images)
      if not all(n == size for n in num_images):
        images_per_class = [ images[:size] for images in images_per_class ]
    # Use first half of images for training, and second half for testing.
    self.train_images = [ images[ : len(images)/2 ]
        for images in images_per_class ]
    self.test_images = [ images[ len(images)/2 : ]
        for images in images_per_class ]

  def SetTrainTestSplitFromDirs(self, train_dir, test_dir, classes = None):
    """Read images from the corpus directories.

    Training and testing subsets are chosen manually.

    :param train_dir: Path to directory of training images.
    :type train_dir: str
    :param test_dir: Path to directory of testing images.
    :type test_dir: str
    :param classes: Class names.
    :type classes: list of str

    .. seealso::
       :func:`SetCorpus`

    """
    if classes == None:
      classes = os.listdir(train_dir)
    train_images = self._ReadCorpusDir(train_dir, classes)
    test_images = self._ReadCorpusDir(test_dir, classes)
    self.SetTrainTestSplit(train_images, test_images, classes)
    self.corpus = (train_dir, test_dir)

  def SetTrainTestSplit(self, train_images, test_images, classes):
    """Manually specify the training and testing images.

    :param train_images: Paths for each training image, with one sub-list per class.
    :type train_images: list of list of str
    :param test_images: Paths for each training image, with one sub-list per class.
    :type test_images: list of list of str
    :param classes: Class names
    :type classes: list of str

    """
    self.classes = classes
    self.train_test_split = 'manual'
    self.train_images = train_images
    self.test_images = test_images

  def ComputeFeatures(self):
    """Compute SVM feature vectors for all images.

    Generally, you do not need to call this method yourself, as it will be
    called automatically by :meth:`RunSvm`.

    """
    if self.train_images == None or self.test_images == None:
      sys.exit("Please specify the corpus.")
    train_sizes = map(len, self.train_images)
    train_size = sum(train_sizes)
    test_sizes = map(len, self.test_images)
    test_size = sum(test_sizes)
    train_images = util.UngroupLists(self.train_images)
    test_images = util.UngroupLists(self.test_images)
    images = train_images + test_images
    # Compute features for all images.
    input_states = [ self.model.MakeStateFromFilename(f, resize = self.resize)
        for f in images ]
    start_time = time.time()
    features = self.ComputeFeaturesFromInputStates(input_states, block = True)
    self.compute_feature_time = time.time() - start_time
    # Split results by training/testing set
    train_features, test_features = util.SplitList(features, [train_size, test_size])
    # Split training set by class
    train_features = util.SplitList(train_features, train_sizes)
    # Split testing set by class
    test_features = util.SplitList(test_features, test_sizes)
    # Store features as list of 2D arrays
    self.train_features = [ np.array(f, util.ACTIVATION_DTYPE)
        for f in train_features ]
    self.test_features = [ np.array(f, util.ACTIVATION_DTYPE)
        for f in test_features ]

  def TrainSvm(self):
    """Construct an SVM classifier from the set of training images.

    :returns: Training accuracy.
    :rtype: float

    """
    if self.train_features == None:
      self.ComputeFeatures()
    start_time = time.time()
    svm_model = svm.ScaledSvm(self.scaler)
    svm_model.Train(self.train_features)
    self.scaler = svm_model.scaler  # scaler has been trained. save it.
    self.classifier = svm_model.classifier
    self.train_results = svm_model.Test(self.train_features)
    return self.train_results['accuracy']

  def TestSvm(self):
    """Test a learned SVM classifier.

    The classifier is applied to the set of test images.

    :returns: Test accuracy.
    :rtype: float

    """
    svm_model = svm.ScaledSvm(classifier = self.classifier,
        scaler = self.scaler)
    self.test_results = svm_model.Test(self.test_features)
    return self.test_results['accuracy']

  def CrossValidateSvm(self):
    """Test a learned SVM classifier.

    The classifier is applied to all images using 10-by-10-way cross-validation.

    :returns: Cross-validation accuracy
    :rtype: float

    """
    test_accuracy = svm.SvmCrossValidate(self.GetFeatures(),
        num_repetitions = 10, num_splits = 10, scaler = self.scaler)
    self.train_results = None
    self.test_results = dict(accuracy = test_accuracy)
    self.cross_validated = True
    return test_accuracy

  def RunSvm(self, cross_validate = False):
    """Train and test an SVM classifier.

    The classifier is trained using the set of training images. Accuracy for
    this classifier is reported for each of the training and testing sets.

    :param bool cross_validate: If true, perform 10x10-way cross-validation.
       Otherwise, compute accuracy for existing training/testing split.
    :returns: Training and testing accuracies. (Training accuracy is None when
       using cross-validation).
    :rtype: 2-tuple of float

    """
    if self.train_features == None:
      self.ComputeFeatures()
    start_time = time.time()
    if cross_validate:
      self.CrossValidateSvm()
      train_accuracy = None
    else:
      self.TrainSvm()
      self.TestSvm()
      train_accuracy = self.train_results['accuracy']
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
    """Load the experiment from disk.

    :returns: The experiment object.
    :rtype: Experiment

    """
    experiment = util.Load(root_path)
    if experiment.classifier != None:
      if not isinstance(root_path, basestring):
        logging.warn("Failed to load SVM model for experiment.")
      else:
        model_path = root_path + '.svm'
        if not os.path.exists(model_path):
          logging.warn("SVM model not found")
        else:
          # Use delayed import of LIBSVM library, so non-SVM methods are always
          # available.
          import svmutil
          experiment.classifier = svmutil.svm_load_model(model_path)
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

  :param model_class: New model class to use (e.g.,
     :class:`viz2.Model <glimpse.models.viz2.model.Model>`).

  """
  global __MODEL_CLASS
  if model_class == None:
    model_class = viz2.Model  # the default GLIMPSE model
  logging.info("Using model type: %s" % model_class.__name__)
  __MODEL_CLASS = model_class
  return __MODEL_CLASS

def GetModelClass():
  """Get the type that will be used to construct an experiment."""
  global __MODEL_CLASS
  if __MODEL_CLASS == None:
    SetModelClass()
  return __MODEL_CLASS

def SetParams(params = None):
  """Set the parameter object that will be used to construct the next
  experiment.

  """
  global __PARAMS
  if params == None:
    params = GetModelClass().ParamsClass()
  __PARAMS = params
  return __PARAMS

def GetParams():
  """Return the parameter object that will be used to construct the next
  experiment.

  """
  global __PARAMS
  if __PARAMS == None:
    SetParams()
  return __PARAMS

def SetLayer(layer = None):
  """Set the layer from which features will be extracted for the next
  experiment.

  """
  global __LAYER, __MODEL_CLASS
  if layer == None:
    layer = GetModelClass().LayerClass.IT
  elif isinstance(layer, str):
    layer = GetModelClass().LayerClass.FromName(layer)
  __LAYER = layer
  return __LAYER

def GetLayer():
  """Return the layer from which features will be extracted for the next
  experiment.

  """
  global __LAYER
  if __LAYER == None:
    SetLayer()
  return __LAYER

def MakeModel(params = None):
  """Create a Glimpse model.

  .. seealso::
     :func:`SetModelClass`

  """
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
  """Manually create a new experiment.

  This function generally is not called directly. Instead, an experiment object
  is implicitly created when needed.

  :param model: The Glimpse model to use for processing images.
  :param layer: The layer activity to use for features vectors.
  :type layer: :class:`LayerSpec <glimpse.models.misc.LayerSpec>` or str
  :param scaler: The feature scaling algorithm, such as the
     :class:`RangeFeatureScaler <glimpse.util.svm.RangeFeatureScaler>`.
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
  if scaler == None:
    scaler = svm.SpheringFeatureScaler()
  __EXP = Experiment(model, layer, pool = GetPool(), scaler = scaler)
  return __EXP

@docstring.copy_dedent(Experiment.ImprintS2Prototypes)
def ImprintS2Prototypes(num_prototypes):
  if __VERBOSE:
    print "Imprinting %d prototypes" % num_prototypes
  result = GetExperiment().ImprintS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

@docstring.copy_dedent(Experiment.MakeUniformRandomS2Prototypes)
def MakeUniformRandomS2Prototypes(num_prototypes):
  if __VERBOSE:
    print "Making %d uniform random prototypes" % num_prototypes
  result = GetExperiment().MakeUniformRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

@docstring.copy_dedent(Experiment.MakeShuffledRandomS2Prototypes)
def MakeShuffledRandomS2Prototypes(num_prototypes):
  if __VERBOSE:
    print "Making %d shuffled random prototypes" % num_prototypes
  result = GetExperiment().MakeShuffledRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

@docstring.copy_dedent(Experiment.MakeHistogramRandomS2Prototypes)
def MakeHistogramRandomS2Prototypes(num_prototypes):
  if __VERBOSE:
    print "Making %d histogram random prototypes" % num_prototypes
  result = GetExperiment().MakeHistogramRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

@docstring.copy_dedent(Experiment.MakeNormalRandomS2Prototypes)
def MakeNormalRandomS2Prototypes(num_prototypes):
  if __VERBOSE:
    print "Making %d normal random prototypes" % num_prototypes
  result = GetExperiment().MakeNormalRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def SetS2Prototypes(prototypes):
  """Set the S2 prototypes from an array or a file.

  :param prototypes: : ndarray of float, str
     The set of prototypes or a path to a file containing the prototypes.

  """
  if isinstance(prototypes, basestring):
    prototypes = util.Load(prototypes)
  elif not isinstance(prototypes, np.ndarray):
    raise ValueError("Please specify an array of prototypes, or the path to a "
        "file.")
  GetExperiment().SetS2Prototypes(prototypes)

@docstring.copy_dedent(Experiment.SetCorpus)
def SetCorpus(corpus_dir, classes = None, balance = False):
  return GetExperiment().SetCorpus(corpus_dir, classes, balance = balance)

@docstring.copy_dedent(Experiment.SetCorpusSubdirs)
def SetCorpusSubdirs(corpus_subdirs, classes = None, balance = False):
  return GetExperiment().SetCorpusSubdirs(corpus_subdirs, classes = classes,
      balance = balance)

@docstring.copy_dedent(Experiment.SetTrainTestSplitFromDirs)
def SetTrainTestSplitFromDirs(train_dir, test_dir, classes = None):
  return GetExperiment().SetTrainTestSplit(train_dir, test_dir, classes)

@docstring.copy_dedent(Experiment.SetTrainTestSplit)
def SetTrainTestSplit(train_images, test_images, classes):
  return GetExperiment().SetTrainTestSplit(train_images, test_images, classes)

@docstring.copy_dedent(Experiment.ComputeFeatures)
def ComputeFeatures():
  GetExperiment().ComputeFeatures()

@docstring.copy_dedent(Experiment.CrossValidateSvm)
def CrossValidateSvm():
  return GetExperiment().CrossValidateSvm()

@docstring.copy_dedent(Experiment.TrainSvm)
def TrainSvm():
  return GetExperiment().TrainSvm()

@docstring.copy_dedent(Experiment.TestSvm)
def TestSvm():
  return GetExperiment().TestSvm()

@docstring.copy_dedent(Experiment.RunSvm)
def RunSvm(cross_validate = False):
  global __VERBOSE
  e = GetExperiment()
  if cross_validate:
    if __VERBOSE:
      print "Computing cross-validated SVM performance on %d images" % \
          sum(map(len, e.GetImages()))
  else:
    if __VERBOSE:
      print "Train SVM on %d images" % sum(map(len, e.train_images))
      print "  and testing on %d images" % sum(map(len, e.test_images))
  train_accuracy, test_accuracy = e.RunSvm(cross_validate)
  if __VERBOSE:
    print "  done: %s s" % e.svm_time
    print "Time to compute feature vectors: %s s" % \
        e.compute_feature_time
    print "Accuracy is %.3f on training set, and %.3f on test set." % \
        (train_accuracy, test_accuracy)
  return train_accuracy, test_accuracy

@docstring.copy_dedent(Experiment.Store)
def StoreExperiment(root_path):
  return GetExperiment().Store(root_path)

@docstring.copy_dedent(Experiment.Load)
def LoadExperiment(root_path):
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
    verbose = 0, resize = None, **opts):
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
  # At this point, all parameters needed to create Experiment object are set.
  GetExperiment().resize = resize
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
      else:
        decision_values = test_results['decision_values']
        print "Decision Values:"
        for cls in range(len(test_images)):
          print "\n".join("%s %s" % _
              for _ in zip(test_images[cls], decision_values[cls]))
    if svm_predicted_labels:
      if 'predicted_labels' not in test_results:
        logging.warn("Predicted labels are unavailable.")
      else:
        predicted_labels = test_results['predicted_labels']
        print "Predicted Labels:"
        for cls in range(len(test_images)):
          print "\n".join("%s %s" % _
              for _ in zip(test_images[cls], predicted_labels[cls]))
  else:
    print "No results available."

def CLIRun(prototypes = None, prototype_algorithm = None, num_prototypes = 10,
    corpus = None, svm = False, compute_features = False, result_path = None,
    cross_validate = False, verbose = 0, balance = False, corpus_subdirs = None,
    **opts):
  if corpus != None:
    SetCorpus(corpus, balance = balance)
  elif corpus_subdirs:  # must be not None and not empty list
    SetCorpusSubdirs(corpus_subdirs, balance = balance)
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
    opts['corpus_subdirs'] = []
    result_path = None
    verbose = 0
    cli_opts, cli_args = util.GetOptions('bc:C:el:m:n:o:p:P:r:R:st:vx',
        ['balance', 'corpus=', 'corpus-subdir=', 'cluster-config=',
        'compute-features', 'edit-options', 'layer=', 'model=',
        'num-prototypes=', 'options=', 'prototype-algorithm=', 'prototypes=',
        'results=', 'resize=', 'svm', 'svm-decision-values',
        'svm-predicted-labels', 'pool-type=', 'verbose', 'cross-validate'])
    for opt, arg in cli_opts:
      if opt in ('-b', '--balance'):
        opts['balance'] = True
      elif opt in ('-c', '--corpus'):
        opts['corpus'] = arg
      elif opt in ('-C', '--corpus-subdir'):
        opts['corpus_subdirs'].append(arg)
      elif opt in ('--cluster-config'):
        # Use a cluster of worker nodes
        opts['cluster_config'] = arg
      elif opt in ('--compute-features'):
        opts['compute_features'] = True
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
      elif opt in ('-R', '--resize'):
        opts['resize'] = int(arg)
      elif opt in ('-s', '--svm'):
        opts['svm'] = True
      elif opt == '--svm-decision-values':
        opts['svm_decision_values'] = True
        opts['verbose'] = max(1, opts['verbose'])
        opts['svm'] = True
      elif opt == '--svm-predicted-labels':
        opts['svm_predicted_labels'] = True
        opts['verbose'] = max(1, opts['verbose'])
        opts['svm'] = True
      elif opt in ('-t', '--pool-type'):
        opts['pool_type'] = arg.lower()
      elif opt in ('-v', '--verbose'):
        opts['verbose'] += 1
      elif opt in ('-x', '--cross-validate'):
        opts['cross_validate'] = True
    CLI(**opts)
  except util.UsageException, e:
    util.Usage("[options]\n"
        "  -b, --balance                   Choose equal number of images per "
        "class\n"
        "  -c, --corpus=DIR                Use corpus directory DIR\n"
        "  -C, --corpus-subdir=DIR         Specify subdirectories (using -C"
        " repeatedly)\n"
        "                                  instead of single corpus directory"
        " (with -c)\n"
        "      --cluster-config=FILE       Read cluster configuration from "
        "FILE\n"
        "      --compute-features          Compute feature vectors (implied "
        "by -s)\n"
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
        "  -R, --resize=NUM                Resize the minimum dimension of "
        "images to NUM\n"
        "                                  pixels\n"
        "  -s, --svm                       Train and test an SVM classifier\n"
        "      --svm-decision-values       Print the pre-thresholded SVM "
        "decision values\n"
        "                                  for each test image (implies -vs)\n"
        "      --svm-predicted-labels      Print the predicted labels for each "
        "test image\n"
        "                                  (implies -vs)\n"
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
