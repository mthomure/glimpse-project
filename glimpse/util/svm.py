"""This module provides easy access to the LIBSVM solver.

Glimpse stores feature vectors as a 3D array-like object. This module includes
functions to convert this data to that required by scikit-learn.

"""

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

def GlimpseToSklearnLabels(features_per_class):
  """Extract scikit-learn labels from Glimpse data.

  :param features_per_class: Per-class feature vectors.
  :type features_per_class: 3D array-like indexed by class, instance, and
     feature.
  :returns: Instance Labels.
  :rtype: 1D-ndarray indexed by instance

  """
  num_classes = len(features_per_class)
  if num_classes == 2:
    labels = [1, -1]
  else:
    labels = range(1, num_classes + 1)
  class_sizes = map(len, features_per_class)
  labels_per_class = [ [label] * size
      for label, size in zip(labels, class_sizes) ]
  # Stack labels vertically. Result is 1D array.
  labels = util.UngroupLists(labels_per_class)
  labels = np.array(labels)
  return labels

def GlimpseToSklearnFeatures(features_per_class):
  """Extract scikit-learn features from Glimpse data.

  :param features_per_class: Per-class feature vectors.
  :type features_per_class: 3D array-like indexed by class, instance, and
     feature.
  :returns: Feature vectors.
  :rtype: 2D-ndarray indexed by instance and feature

  """
  # Stack features vertically. Result is 2D array-like.
  features = util.UngroupLists(features_per_class)
  features = np.array(features)
  return features

def PrepareFeatures(features_per_class):
  """Format feature vectors for use by the sklearn library.

  :param features_per_class: Per-class feature vectors.
  :type features_per_class: 3D array-like indexed by class, instance, and
     feature.
  :returns: Feature vectors and labels.
  :rtype: 2D-ndarray indexed by instance and feature, and 1D-ndarray indexed by
     instance

  """
  return GlimpseToSklearnFeatures(features_per_class), \
      GlimpseToSklearnLabels(features_per_class)

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

  Feature values are "sphered" by subtracting the mean and dividing by the
  standard deviation, as computed on the training set.

  :param train_features: Training data indexed by class, instance, and features.
  :type train_features: 3D array-like
  :rtype: sklearn.base.ClassifierMixin

  """
  # Prepare the data
  labels = GlimpseToSklearnLabels(train_features)
  features = GlimpseToSklearnFeatures(train_features)
  # Create the SVM classifier with feature scaling.
  classifier = Pipeline([ ('scaler', sklearn.preprocessing.Scaler()),
                          ('svm', sklearn.svm.LinearSVC())])
  classifier.fit(features, labels)
  return classifier

def EasyTest(classifier, test_features):
  """
  Apply a built classifier to a set of data.

  :param classifier: A trained classifier, as returned by :func:EasyTrain:.
  :type classifier: sklearn.base.ClassifierMixin
  :param test_features: Testing data indexed by class, instance, and feature.
  :type test_features: 3D array-like
  :rtype: float
  :return: Prediction accuracy of classifier on test data.

  """
  # Prepare the data
  labels = GlimpseToSklearnLabels(test_features)
  features = GlimpseToSklearnFeatures(test_features)
  # Evaluate the classifier
  predicted_labels = classifier.predict(features)
  accuracy = sklearn.metrics.zero_one_score(labels, predicted_labels)
  return accuracy

def EasyTestAUC(classifier, test_features):
  """
  Apply a built classifier to a set of data.

  :param classifier: A trained classifier, as returned by :func:EasyTrain:.
  :type classifier: sklearn.base.ClassifierMixin
  :param test_features: Testing data indexed by class, instance, and feature.
  :type test_features: 3D array-like
  :rtype: float
  :return: Area under the ROC curve of classifier on test data.

  """
  # Prepare the data
  labels = GlimpseToSklearnLabels(test_features)
  features = GlimpseToSklearnFeatures(test_features)
  # Evaluate the classifier
  dvalues = classifier.decision_function(features)
  auc = sklearn.metrics.auc_score(labels, dvalues[:, 0])
  return auc

def EasyCrossVal(classifier, features, num_folds = 5):
  """
  Compute the cross-validated accuracy of an SVM model.

  :param classifier: An untrained classifier.
  :type classifier: sklearn.base.ClassifierMixin
  :param features: Training data indexed by class, instance, and feature.
  :type features: 3D array-like
  :param int num_folds: Number of folds used for cross-validation.
  :rtype: ndarray of float
  :return: Accuracy of model for each fold.

  """
  features, labels = PrepareFeatures(features)
  return sklearn.cross_validation.cross_val_score(classifier, features, labels, cv = num_folds)
