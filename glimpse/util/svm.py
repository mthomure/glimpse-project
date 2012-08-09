"""This module provides access to the LIBSVM solver."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import numpy as np
import sklearn.cross_validation
import sklearn.metrics
import sklearn.pipeline
import sklearn.svm

from glimpse import util

def PrepareFeatures(features_per_class):
  """Format feature vectors for use by the sklearn library.

  :param features_per_class: Per-class feature vectors.
  :type features_per_class: 2D array-like
  :returns: Feature vectors and labels.
  :rtype: 2D-ndarray and 1D-ndarray

  """
  num_classes = len(features_per_class)
  if num_classes == 2:
    labels = [1, -1]
  else:
    labels = range(1, num_classes + 1)
  class_sizes = map(len, features_per_class)
  labels_per_class = [ [label] * size
      for label, size in zip(labels, class_sizes) ]
  features = util.UngroupLists(features_per_class)
  features = np.array(features)
  labels = util.UngroupLists(labels_per_class)
  labels = np.array(labels)
  return features, labels

class Pipeline(sklearn.pipeline.Pipeline):
  """A slightly customized sklearn pipeline, which implements the missing
  decision_function method."""

  def decision_function(self, X):
    """Applies transforms to the data, and the decision_function method of the
    final estimator.

    Valid only if the final estimator implements decision_function.

    """
    Xt = X
    for name, transform in self.steps[:-1]:
        Xt = transform.transform(Xt)
    return self.steps[-1][-1].decision_function(Xt)

def EasyTrain(train_features):
  """
  Train an SVM classifier.

  :type train_features: 3D array-like
  :rtype: sklearn.base.ClassifierMixin

  """
  # Prepare the data
  features, labels = PrepareFeatures(train_features)
  # Create the SVM classifier with feature scaling.
  classifier = Pipeline([ ('scaler', sklearn.preprocessing.Scaler()),
                          ('svm', sklearn.svm.LinearSVC())])
  classifier.fit(features, labels)
  return classifier


def EasyTest(classifier, test_features):
  """
  Apply a built classifier to a set of data.

  :type classifier: sklearn.base.ClassifierMixin
  :type test_features: 3D array-like
  :rtype: float
  :return: Prediction accuracy of classifier on test data.

  """
  # Prepare the data
  features, labels = PrepareFeatures(test_features)
  # Evaluate the classifier
  predicted_labels = classifier.predict(features)
  accuracy = sklearn.metrics.zero_one_score(labels, predicted_labels)
  return accuracy

def EasyCrossVal(classifier, features, num_folds = 5):
  """
  Compute the cross-validated accuracy of an SVM model.

  :type classifier: sklearn.base.ClassifierMixin
  :type features: 3D array-like
  :param int num_folds: Number of folds used for cross-validation.
  :rtype: ndarray of float
  :return: Accuracy of model for each fold.

  """
  features, labels = PrepareFeatures(features)
  return sklearn.cross_validation.cross_val_score(classifier, features, labels, cv = num_folds)
