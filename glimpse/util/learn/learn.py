# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

# Although we do not use PIL in this module, the following import fixes problems
# seen when importing PIL after sklearn on some Mac OS X systems.
import glimpse.util.pil_fix

from collections import Counter
from sklearn.cross_validation import check_cv, check_arrays, is_classifier, \
    Parallel, delayed, clone
from sklearn.metrics import accuracy_score, auc_score
import sklearn.pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np

if hasattr(sklearn.pipeline.Pipeline, 'decision_function'):
  Pipeline = sklearn.pipeline.Pipeline
else:
  # Old sklearn versions have broken Pipeline
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

def cross_val_score(estimator, X, y=None, score_func=None, cv=None, n_jobs=-1,
    verbose=0, as_dvalues=False):
  """Evaluate a score by cross-validation.

  Replacement of :func:`sklearn.cross_validation.cross_val_score`, used to
  support computation of decision values.

  """
  X, y = check_arrays(X, y, sparse_format='csr')
  cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
  if score_func is None:
      if not hasattr(estimator, 'score'):
          raise TypeError(
              "If no score_func is specified, the estimator passed "
              "should have a 'score' method. The estimator %s "
              "does not." % estimator)
  # We clone the estimator to make sure that all the folds are
  # independent, and that it is pickle-able.
  scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
      delayed(_cross_val_score)(clone(estimator), X, y, score_func, train, test,
          verbose, as_dvalues)
      for train, test in cv)
  return np.array(scores)

def _cross_val_score(estimator, X, y, score_func, train, test, verbose,
    as_dvalues):
  """Inner loop for cross validation.

  Modified version of :func:`sklearn.cross_validation._cross_val_score`, used to
  support computation of decision values.

  """
  if y is None:
    estimator.fit(X[train])
    if score_func is None:
      score = estimator.score(X[test])
    else:
      score = score_func(X[test])
  else:
    estimator.fit(X[train], y[train])
    if score_func is None:
      score = estimator.score(X[test], y[test])
    else:
      if as_dvalues:
        predicted_y = estimator.decision_function(X[test])
      else:
        predicted_y = estimator.predict(X[test])
      score = score_func(y[test], predicted_y)
  if verbose > 1:
    print("score: %f" % score)
  return score

def ChooseTrainingSet(labels, train_size):
  """Randomly assign instances to either a training or testing set.

  :param labels: Class labels for instances.
  :type labels: 1D array of int
  :param float train_size: Amount of available data to use (per class) for
     training. If `train_size` < 1, then this is the fraction of available data.
     Otherwise, this is the number of instances (per class) to use for training.
     If fewer instances are available, then all available instances will be used
     for training.
  :rtype: 1D array of bool
  :returns: Mask of training images.

  """
  if train_size < 0:
    raise ValueError("Training size must be positive")
  if train_size >= 1:
    num_labels_for_train = int(train_size)
  if np.isscalar(labels):
    idx = np.arange(labels)
    np.random.shuffle(idx)
    if train_size < 1:
      num_labels_for_train = int(train_size * labels)
    mask = np.zeros((labels,), dtype = np.bool)
    mask[ idx[:num_labels_for_train] ] = True
    return mask
  label_counts = Counter(labels)
  num_instances = len(labels)
  indices = np.arange(num_instances)
  mask = np.zeros((num_instances,), dtype = np.bool)
  for label, count in label_counts.items():
    if count < 2:
      raise ValueError("Need at least two examples of class "
          "(%s) -- (%s) found" % (label, count))
    idx = np.where(labels == label)[0]
    np.random.shuffle(idx)
    if train_size < 1:
      num_labels_for_train = int(train_size * count)
    mask[ idx[:num_labels_for_train] ] = True
  return mask

def ScoreFunctions():
  return 'accuracy', 'auc'

def ResolveScoreFunction(score_func=None):
  return score_func or 'accuracy'

LEARNER_ALIASES = dict(svm='sklearn.svm.LinearSVC',
    logreg='sklearn.linear_model.LogisticRegression')

def ResolveLearner(learner=None):
  if not (isinstance(learner, basestring) or learner is None):
    return learner
  expr = learner or 'svm'  # 'svm' is default algorithm
  expr = LEARNER_ALIASES.get(expr, expr)
  # 'learner' may be an expression. find class name and import.
  if '(' in expr:
    args_idx = expr.index('(')
  else:
    args_idx = len(expr)
  fs = expr[:args_idx].split('.')
  mod_name,short_mod_name,func = '.'.join(fs[:-1]), '.'.join(fs[-2:-1]), fs[-1]
  vs = dict()
  if mod_name:
    import importlib
    try:
      mod = importlib.import_module(mod_name)
      vs[short_mod_name] = mod
      expr = '%s.%s%s' % (short_mod_name, func, expr[args_idx:])
    except ImportError:
      pass
  result = eval(expr, vs)
  if isinstance(result, type):
    result = result()
  return result

def Scaled(algorithm):
  """Create a pipelined algorithm that performs feature scaling."""
  return Pipeline([('scaler', StandardScaler()), ('learner', clone(algorithm))])

def FitClassifier(features, labels, algorithm=None, scale=True):
  """Train a classifier.

  :param algorithm: Type of learning algorithm to apply. Default is a linear-
     kernel SVM.

  """
  if algorithm is None:
    algorithm = LinearSVC()
  if scale:
    algorithm = Scaled(algorithm)
  features = features.astype(float)
  algorithm.fit(features, labels)
  return algorithm

TrainClassifier = FitClassifier  # deprecated

def ScoreClassifier(features, labels, clf=None, score_func=None):
  """Test a learned classifier.

  :type callable score_func: Scoring function (one of accuracy_scorer or
     auc_scorer). This is not a score function from sklearn.metrics.
  :rtype: float
  :returns: Accuracy of classifier on test data.

  """
  # Note: type(clf) will be 'instance' if clf is a learned classifier. If
  # instead type(clf) is 'type', then it is assumed to be the class of learning
  # algorithm to apply.
  if clf is None or type(clf) == type:
    mask = ChooseTrainingSet(labels, 0.5)
    clf = FitClassifier(features[mask], labels[mask], algorithm=clf)
    features = features[~mask]
    labels = labels[~mask]
  features = features.astype(float)
  score_func = score_func or 'accuracy'
  if isinstance(score_func, basestring):
    score_func = score_func.lower()
    if score_func == 'accuracy':
      score_func = accuracy_score
    elif score_func == 'auc':
      predictions = clf.decision_function(features)
      return auc_score(labels, predictions), predictions
  elif not callable(score_func):
    raise ValueError("Score function must be a string or a callable.")
  predictions = clf.predict(features)
  return score_func(labels, predictions), predictions

TestClassifier = ScoreClassifier  # deprecated

def CrossValidateClassifier(features, labels, num_folds=None, algorithm=None,
    scale=True):
  if algorithm is None:
    algorithm = LinearSVC()
  if num_folds is None:
    num_folds = 10
  if scale:
    algorithm = Scaled(algorithm)
  features = features.astype(float)
  return cross_val_score(algorithm, features, labels, cv=num_folds)
