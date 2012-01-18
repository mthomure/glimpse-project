# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

"""This module provides access to the LIBSVM solver."""

import math
from .gio import SuppressStdout
from .misc import GroupIterator, UngroupLists, SplitList
import numpy as np
import os
import sys

class RangeFeatureScaler(object):
  """Scales features to lie in a fixed interval.

  Example Usage:

  instances = np.arange(10).reshape(2, 5)
  scaler = RangeFeatureScaler()
  scaler.Learn(instances)
  scaled_instances = scaler.Apply(instances)
  """

  def __init__(self, min = -1, max = 1):
    """Create new object.
    min -- (float) minimum value in output range
    max -- (float) maximum value in output range
    """
    self.omin, self.omax = min, max

  def Learn(self, features):
    """Determine the parameters required to scale each feature (independently)
    to the range [-1, 1].
    features -- (2D array-like)
    """
    self.imin, self.imax = np.min(features, 0), np.max(features, 0)

  def Apply(self, features):
    """Scale the features in-place. The range of output values will be
    (approximately) [-vmin, vmax], assuming the feature vectors passed here were
    drawn from the same distribution as those used to learn the scaling
    parameters.
    features -- (2D array-like)
    RETURN (2D ndarray) new array of scaled feature values
    """
    features = np.array(features, copy = True)  # copy feature values
    for f in features:
      f -= self.imin  # map to [0, imax - imin]
      f /= (self.imax - self.imin)  # map to [0, 1]
      f *= (self.omax - self.omin)  # map to [0, omax - omin]
      f += self.omin  # map to [omin, omax]
    return features

class SpheringFeatureScaler(object):
  """Scales features to have fixed mean and standard deviation.

  Example Usage:

  instances = np.arange(10).reshape(2, 5)
  scaler = SpheringFeatureScaler()
  scaler.Learn(instances)
  scaled_instances = scaler.Apply(instances)
  """

  def __init__(self, mean = 0, std = 1):
    """Create new object.
    mean -- (float) mean of output feature values
    std -- (float) standard deviation of feature values
    """
    self.omean, self.ostd = mean, std

  def Learn(self, features):
    """Determine the parameters required to scale each feature (independently)
    to the range [-1, 1].
    features -- (2D array-like)
    """
    self.imean, self.istd = np.mean(features, 0), np.std(features, 0)

  def Apply(self, features):
    """Scale the features. The range of output values will be (approximately)
    [-vmin, vmax], assuming the feature vectors passed here were drawn from the
    same distribution as those used to learn the scaling parameters.
    features -- (2D array-like)
    RETURN (2D ndarray) new array with scaled features
    """
    features = np.array(features)  # copy feature values
    features -= self.imean  # map to mean zero
    features /= self.istd  # map to unit variance
    features *= self.ostd  # map to output standard deviation
    features += self.omean  # map to output mean
    return features

def PrepareLibSvmInput(features_per_class):
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
  return UngroupLists(labels_per_class), UngroupLists(features_per_class)

class Svm(object):
  """The LIBSVM classifier."""

  def __init__(self):
    self.classifier = None

  def Train(self, features):
    """Train an SVM classifier.
    features -- (3D array-like) training instances, indexed by class, instance,
                and then feature offset.
    """
    # Use delayed import of LIBSVM library, so non-SVM methods are always
    # available.
    import svmutil
    if not all(isinstance(f, np.ndarray) for f in features):
      raise ValueError("Expected list of arrays for features")
    if not all(f.ndim == 2 for f in features):
      raise ValueError("Expected list of 2D arrays for features")
    svm_labels, svm_features = PrepareLibSvmInput(features)
    options = '-q'  # don't write to stdout
    self.classifier = svmutil.svm_train(svm_labels, svm_features, options)

  def Test(self, features):
    """Test an existing classifier.
    features -- (3D array-like) test instances: indexed by class, instance, and
                then feature offset.
    RETURN (float) accuracy in the range [0, 1]
    """
    # Use delayed import of LIBSVM library, so non-SVM methods are always
    # available.
    import svmutil
    if self.classifier == None:
      raise ValueError("Must train classifier before testing")
    if not all(isinstance(f, np.ndarray) for f in features):
      raise ValueError("Expected list of arrays for features")
    if not all(f.ndim == 2 for f in features):
      raise ValueError("Expected list of 2D arrays for features")
    svm_labels, svm_features = PrepareLibSvmInput(features)
    options = ''  # can't disable writing to stdout
    predicted_labels, acc, decision_values = SuppressStdout(svmutil.svm_predict,
        svm_labels, svm_features, self.classifier, options)
    decision_values = [ dv[0] for dv in decision_values ]
    predicted_labels = SplitList(predicted_labels, map(len, features))
    decision_values = SplitList(decision_values, map(len, features))
    # Ignore mean-squared error and correlation coefficient
    return dict(accuracy = float(acc[0]) / 100.,
        predicted_labels = predicted_labels, decision_values = decision_values)

class ScaledSvm(Svm):
  """A LIBSVM solver, which automatically scales feature values."""

  def __init__(self, scaler = None):
    super(ScaledSvm, self).__init__()
    if scaler == None:
      scaler = SpheringFeatureScaler()
    self.scaler = scaler

  def Train(self, features):
    """Train an SVM classifier.
    features -- (3D array-like) training instances, indexed by class, instance,
                and then feature offset.
    """
    # Learn scaling parameters from single list of all training vectors
    self.scaler.Learn(np.vstack(features))
    # Scale features of training set
    scaled_features = map(self.scaler.Apply, features)
    return super(ScaledSvm, self).Train(scaled_features)

  def Test(self, features):
    """Test an existing classifier.
    features -- (3D array-like) test instances: indexed by class, instance, and
                then feature offset.
    RETURN (float) accuracy in the range [0, 1]
    """
    scaled_features = map(self.scaler.Apply, features)
    return super(ScaledSvm, self).Test(scaled_features)

def SvmForSplit(train_features, test_features, scaler = None):
  """Train and test an SVM classifier from a set of features, which have
  already been partitioned into training and testing sets.
  train_features -- (3D array-like) training instances: indexed by
                    class, instance, and then feature offset.
  test_features -- (list of 2D float ndarray) testing instances: indexed by
                    class, instance, and then feature offset.
  scaler -- feature scaling algorithm
  RETURN (LIBSVM) classifier, (float) training accuracy, (float) test accuracy
  """
  # TEST CASE: unbalanced number of instances across training/testing sets
  # TEST CASE: unbalanced number of instances across classes
  svm = ScaledSvm(scaler)
  svm.Train(train_features)
  return svm.classifier, svm.Test(train_features), svm.Test(test_features)

def SvmCrossValidate(features_per_class, num_repetitions = None,
    num_splits = None, scaler = None):
  """Perform 10x10 way cross-validation.
  features_per_class -- (list of 2D ndarray) feature vectors indexed by class,
                        instance, and then feature offset.
  num_repetitions -- (int) how many times to repeat the measurements (default is
                     10)
  num_splits -- (int) how many ways to split the instances (default is
                num_repetitions)
  scaler -- feature scaling algorithm
  RETURN (float) mean test accuracy across splits and repetitions
  """
  if num_repetitions == None:
    num_repetitions = 10
  elif num_repetitions < 1:
    raise ValueError("Number of repetitions must be positive")
  if num_splits == None:
    num_splits = num_repetitions
  elif num_splits <= 1:
    raise ValueError("Must use at least two splits")
  assert all(isinstance(f, np.ndarray) for f in features_per_class), \
      "Expected sequence of arrays"
  assert all(f.ndim == 2 for f in features_per_class), \
      "Expected sequence of 2D arrays"
  num_classes = len(features_per_class)
  max_num_splits = np.min([f.shape[0] for f in features_per_class])
  assert num_splits <= max_num_splits, \
      "Requested %d splits, but only %d possible" % (num_splits, max_num_splits)

  def make_splits_for_class(instances):
    """Randomize the order of a set of instances for a single class, and split
    them into subsets.
    instances -- (2D array) set of feature vectors
    RETURN (list of 2D array) instance subsets
    """
    # Randomize order of instances
    instances = np.array(instances, copy = True)
    np.random.shuffle(instances)
    # Split into subsets
    split_size, extra = divmod(len(instances), num_splits)
    splits = map(list, GroupIterator(instances, split_size))
    if extra > 0:
      last_bin = UngroupLists(splits[num_splits:])
      splits = splits[:num_splits]
      # Distribute extra instances across remaining splits.
      for split, x in zip(splits, last_bin):
        split.append(x)
    # Convert the set of instances for each split to an array.
    splits = map(np.array, splits)
    assert len(splits) == num_splits, \
        "Unable to create the correct number of splits for all classes: " \
        "requested %d splits, but created %d" % (num_splits, len(splits))
    return splits

  def accuracy_for_split(splits_per_class, test_idx):
    """Compute test accuracy for a given slice, using remaining slices for
    training.
    splits_per_class -- (list of list of ndarray) set of splits for each class,
                        where each split is an matrix of instances.
    test_idx -- (int) index of test split
    """
    # For each class, get all training instances as list of ndarray.
    train_features = [ splits[:test_idx] + splits[test_idx + 1:]
        for splits in splits_per_class ]
    # For each class, combine training instance arrays along first axis
    # (concatenate).
    train_features = map(np.vstack, train_features)
    # For each class, get single array of test instances.
    test_features = [ splits[test_idx] for splits in splits_per_class ]
    svm = ScaledSvm(scaler)
    svm.Train(train_features)
    test_accuracy = svm.Test(test_features)
    return test_accuracy

  def cross_validate():
    # Get accuracy for each slice.
    splits_per_class = map(make_splits_for_class, features_per_class)
    return [ accuracy_for_split(splits_per_class, x)
        for x in range(num_splits) ]

  accuracies = [ cross_validate() for _ in range(num_repetitions) ]
  accuracy = np.mean(accuracies)
  return accuracy
