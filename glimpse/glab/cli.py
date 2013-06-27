"""Provides a command-line interface for running experiments."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

try:
  import matplotlib
  matplotlib.use("cairo")  # workaround old scipy bug
except ImportError:
  pass

import cPickle as pickle
from itertools import chain
import logging
import os
import sys

from glimpse.experiment import *
from glimpse.experiment.prototype_algorithms import (GetAlgorithmNames,
    ResolveAlgorithm)
from glimpse.models import MakeModel
from glimpse.pools import MakePool
from glimpse.util.option import *
from glimpse.util.learn import ScoreFunctions, ResolveLearner
from glimpse.util.progress import ProgressBar

def MakeCliOptions():
  """Options for the command-line project"""
  return OptionRoot(
        Option('verbose', False, flag = ('v', 'verbose'),
            doc = "Enable verbose logging"),
        Option('input_path', flag = ('i:', 'input='),
            doc = "Read initial experiment data from a file"),
        Option('result_path', flag = ('o:', 'output='),
            doc = "Store results to a file"),
        Option('pool_type', None, flag = ('t:', 'pool-type='),
            enum = ('s', 'singlecore', 'm', 'multicore', 'c', 'cluster'),
            doc = "Set the worker pool type"),
        Option('train_size', None, flag = ('T:', 'train-size='),
            doc = "Set the size of the training set (number of instances or "
                "fraction of total)"),
        Option('timing', flag = ('timing'), doc = "Report timing"
            " information for worker pool (assumes cluster pool is used)"),
        Option('command', flag = ('command='), doc = "Execute a command "
            "after running the experiment (but before results are saved"),
        OptionGroup('corpus',
            Option('root_dir', flag = ('c:','corpus='),
                doc = "Set corpus directory"),
            Option('subdirs', flag = ('C:', 'corpus-subdir='), multiple=True,
                doc = "Specify subdirectories (using -C repeatedly)"),
            Option('from_name', flag = 'corpus-name=',
                doc = "Specify corpus by name (one of 'easy', 'moderate', or "
                    "'hard')"),
            Option('balance', False, flag = ('b', 'balance'),
                doc = "Choose equal number of images per class")),
        OptionGroup('extractor',
            Option('param_path', flag = ('O:', 'options='),
                doc = "Read model options from a file"),
            Option('no_activity', flag = ('N', 'no-activity'),
                doc = "Do not compute activity model activity for each image "
                    "(implies no classifier). This can be used to learn "
                    "prototypes without immediately evaluating them."),
            Option('save_all', flag = ('A', 'save-all'),
                doc = "Save activity for all layers, rather than just the "
                    "layers from which features are extracted."),
            OptionGroup('prototypes',
                Option('path', flag = ('P:', 'prototypes='),
                    doc = "Read S2 prototypes from a file (overrides -p)"),
                Option('length', default = 10, flag = ('n:', 'num-prototypes='),
                    doc = "Number of S2 prototypes to generate"),
                Option('algorithm', flag = ('p:', 'prototype-algorithm='),
                    enum = sorted(GetAlgorithmNames()),
                    doc = "Specify how S2 prototypes are generated"),
                Option('low', flag = 'low=', default = 0., doc = "Low end "
                    "of random prototype distribution"),
                Option('high', flag = 'high=', default = 1., doc = "High "
                    "end of random prototype distribution"),
                Option('num_samples', flag = 'samples=', default=0,
                    doc = "Number of training patches to sample for S2 "
                    "prototype learning"),
                Option('num_regr_samples', flag='regr-samples=',
                    default=0, doc="Number of patches to sample when training "
                    "regression model for meta_feature_wkmeans"),
                Option('mask_dir', flag='masks=', default='',
                    doc="Mask directory for object_mask_wkmeans"),
                Option('base_weight', flag='base-weight=', default=0.,
                    doc="Value added to weight for all training patches"),
                )),
        OptionGroup('evaluation',
            Option('layer', default = "C2", flag = ('l:', 'layer='),
                doc = "Choose the layer(s) from which features are extracted"),
            Option('evaluate', default = False, flag = ('E', 'evaluate'),
                doc = "Train and test a classifier"),
            Option('cross_validate', default = False, flag = ('x',
                'cross-validate'), doc = "Compute test accuracy via (10x10-way)"
                    "cross-validation instead of fixed training/testing split"),
            Option('cross_val_folds', default = 10, flag = ('f:',
                'num-folds='), doc = "Number of folds for cross-validation"),
            Option('score_func', "accuracy", flag = ('S:', 'score-function='),
                   enum = ScoreFunctions(),
                   doc = "Specify the scoring function for classifier "
                      "evaluation"),
            Option('hist_features', False, flag = ('H', 'hist-features'),
                doc = "Use histograms (accumulated over space and scale) for "
                    "each feature band (requires spatial features, such as "
                    "C1)"),
            Option('learner', 'svm', flag = ('L:', 'learner='),
                   doc = "Learning algorithm to use for classification (can be "
                      "a Python expression, or one of 'svm' or 'logreg')"),
            ),
        Option('help', flag = ('h', 'help'), doc = "Print this help and exit"))

def CliWithActivity(opts, exp, pool):
  progress = None
  if opts.verbose:
    progress = ProgressBar
  # Initialize model
  if exp.extractor.model is None:
    if opts.extractor.param_path:
      logging.info("Reading model parameters from file: %s" %
          opts.extractor.param_path)
      with open(opts.extractor.param_path) as fh:
        params = pickle.load(fh)
    else:
      params = None
    exp.extractor.model = MakeModel(params)
  elif opts.extractor.param_path:
    logging.warn("Ignoring model parameter file (model exists)!")
  # Initialize prototypes
  popts = opts.extractor.prototypes
  if popts.path:
    with open(popts.path) as fh:
      protos = pickle.load(fh)
    # XXX assuming single-size prototypes
    logging.info("Read %d prototypes from file: %s" % (len(protos[0]),
        popts.path))
    exp.extractor.model.s2_kernels = protos
  elif popts.algorithm:
    num_prototypes = int(popts.length)
    alg = ResolveAlgorithm(popts.algorithm)
    alg = alg()  # create algorithm object of given class
    for key in ('low', 'high', 'num_samples', 'num_regr_samples', 'mask_dir',
        'base_weight'):
      if hasattr(alg, key):
        setattr(alg, key, getattr(popts, key))
    MakePrototypes(exp, num_prototypes, alg, pool, train_size=opts.train_size,
        progress=progress)
  # Compute model activation
  if not opts.extractor.no_activity:
    report_timing = opts.timing and hasattr(pool, 'save_timing')
    if report_timing:
      pool.save_timing = True
    ComputeActivation(exp, opts.evaluation.layer, pool,
        save_all = opts.extractor.save_all, progress = progress)
    if report_timing:
      exp.extractor['timing'] = pool.timing
      if opts.verbose:
        import importlib
        m = importlib.import_module(pool.__module__)
        print "START TIMING REPORT"
        m.PrintTiming(pool.timing)
        print "END TIMING REPORT"

# for large datasets, it's useful to stream results to disk as they arrive,
# rather than wait for the entire batch to complete. in this case we must
#  1) open file handle fh
#  3) for each batch of exp.corpus.paths:
#    a) compute activation
#    b) store to fh
#  4) close fh
# note that activation is stored in the same order as that of corpus.paths.
# afterwards, activation can be read in and stored in the experiment.

# to support this, we could pass a pool wrapper, which chunks the input
# arguments and delegates to original pool.map() for each chunk. [how does this
# interact chunksize argument in ipythonpool.map?]

class TeePool(object):
  """A pool object that incrementally processes results as they arrive."""
  chunksize = None  # size of each batch

  def __init__(self, pool):
    self.pool = pool

  def map(self, func, iterable, progress=None, chunksize=None):
    # XXX progress bar is currently unsupported
    return list(iterator.chain(self.process(self.pool.map(func, batch))
        for batch in batches))

  def process(self, results):
    """Perform processing on incremental results.

    :rtype: iterable
    :return: Incremental results as returned to caller. Return an empty list to
       remove all results.

    """
    return results

def CliEvaluate(opts, exp):
  if not opts.evaluation.layer:
    raise OptionError("Must specify model layer to use for features")
  if opts.evaluation.hist_features:
    feature_builder = ExtractHistogramFeatures
  else:
    feature_builder = None
  try:
    learner = ResolveLearner(opts.evaluation.learner)
  except Exception, e:
    raise OptionError("Error in 'learner' expression -- %s" % e)
  if opts.evaluation.cross_validate:
    CrossValidateClassifier(exp, opts.evaluation.layer,
        feature_builder = feature_builder,
        num_folds = opts.evaluation.cross_val_folds,
        score_func = opts.evaluation.score_func,
        learner = learner)
  else:
    TrainAndTestClassifier(exp, opts.evaluation.layer,
        train_size = opts.train_size, feature_builder = feature_builder,
        score_func = opts.evaluation.score_func,
        learner = learner)

def CliProject(opts):
  # Read verbosity from environment var unless flag is given.
  log_level = Verbose(opts.verbose or None)
  if (log_level != logging.INFO and opts.result_path is None and
      opts.command is None):
    logging.warn("No results will be given. You probably want to specify a "
        "results file or command, or enable the verbose flag.")
  reader = DirReader(ignore_hidden = True)
  # Initialize experiment object
  if opts.input_path:
    # Read experiment from disk
    logging.info("Reading initial experiment data from file -- %s",
        opts.input_path)
    if opts.input_path == '-':
      exp = pickle.load(sys.stdin)
    else:
      with open(opts.input_path, 'rb') as fh:
        exp = pickle.load(fh)
  else:
    exp = ExperimentData()
  # Initialize corpus: Each sub-directory is given a distinct numeric label,
  # starting at one. If the root directory is given, labels are assigned to
  # sub-directories in alphabetical order.
  if opts.corpus.subdirs:
    SetCorpusSubdirs(exp, opts.corpus.subdirs, opts.corpus.balance, reader)
  else:
    path = None
    if opts.corpus.root_dir:
      path = opts.corpus.root_dir
    elif opts.corpus.from_name:
      path = GetCorpusByName(opts.corpus.from_name)
    if path:
      SetCorpus(exp, path, opts.corpus.balance, reader)
    elif not exp.corpus.paths:
      raise OptionError("Must specify a corpus")
  # Initialize model and compute activation, if necessary.
  eopts = opts.extractor
  pool = None
  if (not eopts.no_activity or eopts.prototypes.path or
      eopts.prototypes.algorithm):
    pool = MakePool(opts.pool_type)
    logging.info("Using pool: %s" % type(pool).__name__)
    CliWithActivity(opts, exp, pool)
  # Evaluate features
  if opts.evaluation.evaluate:
    CliEvaluate(opts, exp)
  if opts.command:
    exec opts.command
  # Store experiment to disk
  if opts.result_path == '-':
    logging.info("Writing experiment data to stdout")
    pickle.dump(exp, sys.stdout, protocol = 2)
  elif opts.result_path is not None:
    logging.info("Writing experiment data to file -- %s" % opts.result_path)
    with open(opts.result_path, 'wb') as fh:
      pickle.dump(exp, fh, protocol = 2)
  return exp

def Main(argv = None):
  options = MakeCliOptions()
  try:
    ParseCommandLine(options, argv = argv)
    if options.help.value:
      print >>sys.stderr, "Usage: [options]"
      PrintUsage(options)
      sys.exit(-1)
    CliProject(OptValue(options))
  except ExpError, e:
    print >>sys.stderr, "Error: %s." % e
  except OptionError, e:
    print >>sys.stderr, "Usage Error (use -h for help): %s." % e

if __name__ == '__main__':
  Main()

"""
== Example use cases that should be supported (at least for the gui) ==

Given classification results, view false positive or false negative images.

"""
