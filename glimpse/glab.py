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
SetExperiment(Experiment(layer = 'C1'))
SetCorpus("indir/corpus")

# If we had wanted to configure the parameters of the Glimpse model, we could
# have constructed the model manually when initializing the experiment.
params = Params()
params.num_scales = 8
SetExperiment(Experiment(model = Model(params)))
SetCorpus("indir/corpus")
"""

from glimpse import backends
from glimpse.models import viz2
from glimpse import util
import itertools
import numpy as np
import operator
import os
import sys

__all__ = ( 'Experiment', 'GetExperiment', 'SetExperiment',
    'ImprintS2Prototypes', 'SetCorpus', 'SetTrainTestSplit', 'TrainSvm',
    'TestSvm', 'LoadExperiment', 'StoreExperiment', 'Params', 'Model' )

def ChainLists(*iterables):
  """Concatenate several sequences to form a single list."""
  return list(itertools.chain(*iterables))

#~ class SinglecorePool(object):
#~
  #~ def map(self, func, iterable, chunksize = None):
    #~ return map(func, iterable)
#~
  #~ def imap(self, func, iterable, chunksize = 1):
    #~ return map(func, iterable)
#~
  #~ def imap_unordered(self, func, iterable, chunksize = 1):
    #~ return map(func, iterable)
#~
#~ class MulticorePool(object):
  #~ """Thin wrapper around multiprocessing.Pool that supports serialization."""
#~
  #~ def __init__(self, *args):
    #~ """Create new object. See multiprocessing.Pool() for explanation of
    #~ arguments."""
    #~ # Save the initialization arguments, so we can serialize and then
    #~ # reconstruct this object later.
    #~ self._init_args = args
    #~ import multiprocessing
    #~ self.pool = multiprocessing.Pool(*args)
#~
  #~ def __reduce__(self):
    #~ return (MulticorePool, self._init_args)
#~
  #~ def map(self, func, iterable, chunksize = None):
    #~ return self.pool.map(func, iterable, chunksize)
#~
  #~ def imap(self, func, iterable, chunksize = 1):
    #~ return self.pool.imap(func, iterable, chunksize)
#~
  #~ def imap_unordered(self, func, iterable, chunksize = 1):
    #~ return self.pool.imap_unordered(func, iterable, chunksize)

class Experiment(object):

  def __init__(self, model = None, pool = None, layer = None):
    """Create a new experiment.
    model -- the Glimpse model to use for processing images
    pool -- a serializable worker pool (default is a MulticorePool)
    layer -- (LayerSpec or str) the layer activity to use for features vectors
    """
    if model == None:
      model = Model()
    if pool == None:
      pool = MulticorePool()
    if layer == None:
      layer = model.Layer.IT
    elif isinstance(layer, str):
      layer = model.Layer.FromName(layer)
    self.model = model
    self.pool = pool
    self.layer = layer
    # Initialize attributes used by an experiment
    self.classifier = None
    self.corpus = None
    self.prototype_source = None
    self.test_error = None
    self.test_images = None
    self.train_error = None
    self.train_images = None
    self.train_test_split = None

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
    """Imprint a set of S2 prototypes from a set of training images."""
    if self.train_images == None:
      sys.exit("Please specify the training corpus before calling "
          "ImprintS2Prototypes().")
    image_files = ChainLists(*self.train_images)
    patches_per_image, extra = divmod(num_prototypes, len(image_files))
    if extra > 0:
      patches_per_image += 1
    sampler = self.model.C1PatchSampler(patches_per_image, normalize = True)
    # Represent each image file as an empty model state.
    input_states = map(self.model.MakeStateFromFilename, image_files)
    # Compute C1 activity, and sample patches.
    values_per_image = self.pool.imap_unordered(sampler, input_states)
    # Concatenate values from each image.
    all_values = list(itertools.chain(*values_per_image))
    # Ignore the imprint locations.
    all_prototypes = map(operator.itemgetter(0), all_values)
    # We may have sampled too many C1 patches, so truncate the list.
    all_prototypes = all_prototypes[:num_prototypes]
    # Convert to numpy array.
    all_prototypes = np.array(all_prototypes, util.ACTIVATION_DTYPE)
    # Store new prototypes in model.
    self.prototype_source = 'imprinted'
    self.model.s2_kernels = all_prototypes

  def ComputeFeatures(self, input_states):
    """Return the activity of the model's output layer for a set of images.
    input_states -- (State iterable) model states containing image data
    RETURN (iterable) a feature vector for each image
    """
    L = self.model.Layer
    if self.layer in (L.S2, L.C2, L.IT) and self.model.s2_kernels == None:
      sys.exit("Please set the S2 prototypes before computing feature vectors "
          "for layer %s." % self.layer.name)
    transform = self.model.ModelTransform(self.layer, save_all = False)
    # Compute model states containing IT features.
    output_states = self.pool.map(transform, input_states)
    # Look up the activity values for the output layer, and convert them all to
    # a single vector.
    return [ util.ArrayListToVector(state[self.layer.id].activity)
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
               the SVM classes. In the case of a binary SVM, the first class is
               "positive", while the second is "negative".
    """
    if classes == None:
      classes = os.listdir(corpus_dir)
    self.classes = classes
    corpus_dir = os.path.abspath(corpus_dir)
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
    train_dir = os.path.abspath(train_dir)
    test_dir = os.path.abspath(test_dir)
    self.corpus = (train_dir, test_dir)
    self.train_test_split = 'manual'
    self.train_images = self._ReadCorpusDir(train_dir, classes)
    self.test_images = self._ReadCorpusDir(test_dir, classes)

  def _ComputeLibSvmFeatures(self, multiclass_images):
    """Internal helper function for TrainSvm() and TestSvm()."""
    # Compute SVM features for all images.
    all_images = ChainLists(*multiclass_images)
    # Represent each image file as an empty model state.
    input_states = map(self.model.MakeStateFromFilename, all_images)
    features = self.ComputeFeatures(input_states)
    # Break apart feature vectors for positive and negative instances.
    pos_features = features[ : len(multiclass_images[0]) ]
    neg_features = features[ len(multiclass_images[0]) : ]
    classes = [1] * len(pos_features) + [-1] * len(neg_features)
    features = pos_features + neg_features
    # Convert feature vectors from np.array to list objects
    features = map(list, features)
    return classes, features

  def TrainSvm(self):
    """Train an SVM classifier from the set of training images."""
    if self.train_images == None:
      sys.exit("Please specify the training corpus before calling TrainSvm().")
    classes, features = self._ComputeLibSvmFeatures(self.train_images)
    options = '-q'  # don't write to stdout
    # Use delayed import of LIBSVM library, so non-SVM methods are always
    # available.
    import svmutil
    self.classifier = svmutil.svm_train(classes, features, options)
    options = ''  # can't disable writing to stdout
    predicted_labels, acc, decision_values = svmutil.svm_predict(classes,
        features, self.classifier, options)
    self.train_error = 1. - float(acc[0]) / 100.
    return self.train_error

  def TestSvm(self):
    """Apply the classifier to a set of test images.
    RETURN (float) accuracy
    """
    if self.classifier == None:
      sys.exit("Please train the classifier before calling TestSvm().")
    if self.test_images == None:
      sys.exit("Please specify the testing corpus before calling TestSvm().")
    classes, features = self._ComputeLibSvmFeatures(self.test_images)
    options = ''  # can't disable writing to stdout
    # Use delayed import of LIBSVM library, so non-SVM methods are always
    # available.
    import svmutil
    predicted_labels, acc, decision_values = svmutil.svm_predict(classes,
        features, self.classifier, options)
    # Ignore mean-squared error and correlation coefficient
    self.test_error = 1. - float(acc[0]) / 100.
    return self.test_error

  def Store(self, root_path):
    """Save the experiment to disk."""
    # We modify the value of the "classifier" attribute, so cache it.
    classifier = self.classifier
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
    # Restore the value of the "classifier" attribute.
    self.classifier = classifier

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

__EXP = None

def GetExperiment():
  global __EXP
  if __EXP == None:
    __EXP = Experiment()
  return __EXP

def SetExperiment(experiment):
  global __EXP
  __EXP = experiment

def ImprintS2Prototypes(num_prototypes):
  return GetExperiment().ImprintS2Prototypes(num_prototypes)

def SetCorpus(corpus_dir, classes = None):
  return GetExperiment().SetCorpus(corpus_dir, classes)

def SetTrainTestSplit(train_dir, test_dir, classes = None):
  return GetExperiment().SetTrainTestSplit(train_dir, test_dir, classes)

def TrainSvm():
  return GetExperiment().TrainSvm()

def TestSvm():
  return GetExperiment().TestSvm()

def LoadExperiment(root_path):
  global __EXP
  __EXP = Experiment.Load(root_path)
  return __EXP

def StoreExperiment(root_path):
  GetExperiment().Store(root_path)

Params = viz2.Params

def Model(params = None):
  """Create the default model."""
  if params == None:
    params = viz2.Params()
  return viz2.Model(backends.CythonBackend(), params)
