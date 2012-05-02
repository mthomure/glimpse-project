"""Functions related to computing statistics on a set of data."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import numpy

def Pca(X):
  """Compute the Principal Component Analysis (PCA) transformation for a
     dataset.

  The first k rows of the transformation correspond to a projection onto a
  k-dimensional surface, chosen such that the L2 approximation error on the
  training set is minimized. This transformation is given as:

  .. math::
    y = A'(x - \mu),

  where :math:`\mu` is the mean of the training data, and A has columns given by
  the eigenvectors of the training data's covariance matrix. Eigenvectors are
  sorted by descending eigenvalue, which provides that the first k rows of the
  transformation are the optimal linear transformation under L2 approximation
  error. Returns the transform, and the standard deviation for each axis of the
  training data.

  Usage:

     >>> T, S = Pca(X)

  where X is a matrix of training data, T is the transformation matrix, S is the
  array of standard devations. To transform a data point W given as an array,
  use::

     >>> Y = numpy.dot(T, W)

  :param X: Input data, with variables given by columns and observations given
     by rows.
  :type X: 2D ndarray of float

  .. seealso::
     This function was adapted from `similar code by Jan Erik Solem
     <http://www.janeriksolem.net/2009/01/pca-for-images-using-python.html>`_.
     Also see :func:`sklearn.decomposition.PCA`.

  """
  if len(X.shape) != 2:
    raise Exception("Training data must be a matrix")
  mean = X.mean(0)
  X = X - mean
  # Find covariance matrix of X - mu
  cov = numpy.dot(X, X.T)
  # Find eigenvectors of symmetric covariance matrix
  eigenvalues, eigenvectors = numpy.linalg.eigh(cov)
  # Full transformation
  transform = numpy.dot(X.T, eigenvectors).T
  # Reorder transformation by descending eigenvalue.
  order = numpy.argsort(eigenvalues)[::-1]
  transform = transform[ order ]
  # Any negative eigenvalues are zero, and negative sign is caused by numerical
  # approximation error.
  eigenvalues[ eigenvalues < 0 ] = 0
  stdev = numpy.sqrt(eigenvalues)[ order ]
  return transform, stdev

def CalculateRoc(target_labels, predicted_labels):
  """Calculate the points of the ROC curve from a set of labels and evaluations
  of a classifier.

  Uses the single-pass efficient algorithm of Fawcett (2006). This assumes a
  binary classification task.

  :param target_labels: Ground-truth label for each instance.
  :type target_labels: 1D ndarray of float
  :param predicted_labels: Predicted label for each instance.
  :type predicted_labels: 1D ndarray of float
  :returns: Points on the ROC curve
  :rtype: 1D ndarray of float

  .. seealso::
     :func:`sklearn.metrics.roc_curve`

  """
  def iterator():
    num_pos = len(target_labels[ target_labels == 1 ])
    num_neg = len(target_labels) - num_pos
    i = predicted_labels.argsort()[::-1]
    fp = tp = 0
    last_e = -numpy.inf
    for l, e in zip(target_labels[i], predicted_labels[i]):
      if e != last_e:
        yield (fp / float(num_neg), tp / float(num_pos))
        last_e = e
      if l == 1:
        tp += 1
      else:
        fp += 1
    yield (fp / float(num_neg), tp / float(num_pos))
  return numpy.array(list(iterator()))

def CalculateRocScore(target_labels, predicted_labels):
  """Calculate area under the ROC curve (AUC) from a set of target labels and
  predicted labels of a classifier.

  :param target_labels: Ground-truth label for each instance.
  :type target_labels: 1D ndarray of float
  :param predicted_labels: Predicted label for each instance.
  :type predicted_labels: 1D ndarray of float
  :returns: AUC value
  :rtype: float

  .. seealso::
     :func:`sklearn.metrics.auc`

  """
  import scipy.integrate
  points = CalculateRoc(target_labels, predicted_labels)
  p = points.T
  return scipy.integrate.trapz(p[1], p[0]), points

def DPrime(true_positive_rate, false_positive_rate):
  """Calculate discriminability (d') measure.

  :param float true_positive_rate: Normalized TP rate.
  :param float false_positive_rate: Normalized FP rate.
  :rtype: float

  """
  # WARN: scipy.stats not available by default on darwin
  from scipy.stats import norm
  # Lower bound on TP/FP is 0.001%
  # Upper bound is 99.999%
  reg = 0.00001
  true_positive_rate = max(min(true_positive_rate, 1.0 - reg), reg)
  false_positive_rate = max(min(false_positive_rate, 1.0 - reg), reg)
  a = norm.ppf(true_positive_rate)
  b = norm.ppf(false_positive_rate)
  return a - b
