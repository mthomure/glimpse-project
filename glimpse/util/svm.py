"""This module provides access to the LIBSVM solver."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import numpy as np

from glimpse import util
import sklearn.pipeline

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
