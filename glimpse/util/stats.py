
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Functions related to computing statistics on a set of data.
#

import numpy

# Adapted from:
#   http://www.janeriksolem.net/2009/01/pca-for-images-using-python.html
def Pca(X):
  """Compute the Principal Component Analysis (PCA) transformation for a dataset
  given as a 2D matrix. The first k rows of the transformation correspond to a
  projection onto a k-dimensional surface, chosen such that the L2 approximation
  error on the training set is minimized. This transformation is given as
    y = A'(x - mu),
  where mu is the mean of the training data, and A has columns given by the
  eigenvectors of the training data's covariance matrix. Eigenvectors are sorted
  by descending eigenvalue, which provides that the first k rows of the
  transformation are the optimal linear transformation under L2 approximation
  error. Returns the transform, the standard deviation for each axis, and the mean of
  the training data.
  Usage:
    T, S, M = Pca(X)
  where X is a matrix of training data, T is the transformation matrix, S is the
  array of standard devations, and M is the mean of the training data. To
  transform a data point W given as an array, use
    Y = numpy.dot(T, W)
  """
  if len(X.shape) != 2:
    raise Exception("Training data must be a matrix")
  mean = X.mean(0)
  X = X - mean
  # Find covariance matrix X - mu
  cov = numpy.dot(X, X.T)
  # Find eigenvectors of symmetric covariance matrix
  eigenvalues, eigenvectors = numpy.linalg.eigh(cov)
  # Full transformation
  transform = numpy.dot(X.T, eigenvectors).T
  # Reorder transformation by descending eigenvalue.
  order = numpy.argsort(eigenvalues)[::-1]
  transform = transform[ order ]
  stdev = numpy.sqrt(eigenvalues)[ order ]
  mean = mean[ order ]
  return transform, stdev, mean

def CalculateRoc(labels, evaluations):
  """Calculate the points of the ROC curve from a set of labels and evaluations
  of a classifier. Uses the single-pass efficient algorithm of Fawcett
  (2006)."""
  def iterator():
    num_pos = len(labels[ labels == 1 ])
    num_neg = len(labels) - num_pos
    i = evaluations.argsort()[::-1]
    fp = tp = 0
    last_e = -numpy.inf
    for l, e in zip(labels[i], evaluations[i]):
      if e != last_e:
        yield (fp / float(num_neg), tp / float(num_pos))
        last_e = e
      if l == 1:
        tp += 1
      else:
        fp += 1
    yield (fp / float(num_neg), tp / float(num_pos))
  return numpy.array(list(iterator()))

def CalculateRocScore(labels, evaluations):
  """Calculate area under the ROC curve from a set of labels and evaluations of
  a classifier."""
  import scipy.integrate
  points = CalculateRoc(labels, evaluations)
  p = points.T
  return scipy.integrate.trapz(p[1], p[0]), points

def DPrime(true_positive_rate, false_positive_rate):
  """Calculate discriminability measure, given normalized true positive and
  false positive counts."""
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
