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
import pprint
import sys

from glimpse.experiment import *
from glimpse.experiment.prototype_algorithms import (GetAlgorithmNames,
    ResolveAlgorithm)
from glimpse.models import MakeModel, MakeParams
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
            doc = "Set the worker pool type. Can also use the 'GLIMPSE_POOL' "
                "environment variable. If using a cluster pool type, the "
                "cluster package and arguments are read from the "
                "GLIMPSE_CLUSTER and 'GLIMPSE_CLUSTER_ARGS' environment "
                "variables."),
        Option('train_size', None, flag = ('T:', 'train-size='),
            doc = "Set the size of the training set (number of instances or "
                "fraction of total)"),
        Option('timing', flag = ('timing'), doc = "Report timing"
            " information for worker pool (assumes cluster pool is used)"),
        Option('command', flag = ('command='), doc = "Execute a command "
            "after running the experiment (but before results are saved)"),
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
            Option('param_file', flag = ('param-file='),
                doc = "Read model options from a file"),
            Option('params', flag = ('P:', 'param='), multiple=True,
                doc = "Set model options from command line (e.g.: glab -P "
                    "num_scales=3 -P scale_factor=1.25"),
            Option('no_activity', flag = ('N', 'no-activity'),
                doc = "Do not compute activity model activity for each image "
                    "(implies no classifier). This can be used to learn "
                    "prototypes without immediately evaluating them."),
            Option('save_all', flag = ('A', 'save-all'),
                doc = "Save activity for all layers, rather than just the "
                    "layers from which features are extracted."),
            OptionGroup('prototypes',
                Option('path', flag = ('prototype-file='),
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
                'cross-validate'), doc = "Compute test accuracy via "
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
            Option('predictions', flag = 'predictions', doc = "Print the true "
                "and predicted labels for each image in the corpus (only for "
                "fixed-split evaluations)."),
            ),
        Option('help', flag = ('h', 'help'), doc = "Print this help and exit"))

def InitModel(opts, exp):
  if exp.extractor.model is None:
    if opts.extractor.param_file:
      logging.info("Reading model parameters from file: %s" %
          opts.extractor.param_file)
      with open(opts.extractor.param_file) as fh:
        params = pickle.load(fh)
    else:
      params = MakeParams()
    if opts.extractor.params:
      args = [p.split('=') for p in opts.extractor.params]
      if not all(len(a) == 2 for a in args):
        raise OptionError("Must specify model parameters as KEY=VALUE")
      for k,v in args:
        setattr(params, k, type(params.trait(k).default)(v))
    exp.extractor.model = MakeModel(params)
  elif opts.extractor.param_file or opts.extractor.params:
    logging.warn("Ignoring user's model parameters (model exists)!")

def CliWithActivity(opts, exp, pool):
  progress = None
  if opts.verbose:
    progress = ProgressBar
  InitModel(opts, exp)
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

def CheckClassLabels(exp):
  if exp.corpus.labels is None:
    return
  labels = np.unique(exp.corpus.labels)
  labels.sort()
  if len(labels) != 2 or tuple(labels) == (0,1):
    return
  logging.warning("Found binary classification task without zero-one labels. "
      "Fixing.")
  idx_0 = exp.corpus.labels == labels[0]
  idx_1 = exp.corpus.labels == labels[1]
  exp.corpus.labels[idx_0] = 0
  exp.corpus.labels[idx_1] = 1

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
    if opts.evaluation.score_func not in (None, 'accuracy'):
      logging.warn("Ignoring score_func of '%s'. Cross-validation always uses "
          "'accuracy'.", opts.evaluation.score_func)
    CrossValidateClassifier(exp, opts.evaluation.layer, learner=learner,
        feature_builder=feature_builder,
        num_folds=opts.evaluation.cross_val_folds)
  else:
    TrainAndTestClassifier(exp, opts.evaluation.layer,
        train_size=opts.train_size, feature_builder=feature_builder,
        score_func=opts.evaluation.score_func, learner=learner)
    if opts.evaluation.predictions:
      print
      print "Classifier Predictions"
      print "======================"
      print
      print ("Each line gives the true and predicted labels (in that order) "
            "for an image in the corpus.")
      print
      print "Training Set Predictions"
      print "------------------------"
      training_predictions = GetPredictions(exp, training=True, evaluation=-1)
      if len(training_predictions) == 0:
        print "no training instances"
      else:
        for img,lbl,pred in training_predictions:
          print img, lbl, pred
      print
      print "Test Set Predictions"
      print "--------------------"
      predictions = GetPredictions(exp, training=False, evaluation=-1)
      if len(predictions) == 0:
        print "no test instances"
      else:
        for img,lbl,pred in predictions:
          print img, lbl, pred
      print

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
    CheckClassLabels(exp)
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
    elif exp.corpus.paths is None:
      raise OptionError("Must specify a corpus")
  # Initialize model and compute activation, if necessary.
  eopts = opts.extractor
  pool = None
  if (not eopts.no_activity or eopts.prototypes.path or
      eopts.prototypes.algorithm):
    pool = MakePool(opts.pool_type)
    logging.info("Using pool: %s" % type(pool).__name__)
    CliWithActivity(opts, exp, pool)
  elif eopts.param_file or eopts.params:
    # Ensure model is created if parameters are set.
    InitModel(opts, exp)
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

import textwrap

def PrintDict(data, max_key_len=25, width=None, stream=None, pad=3):
  if stream is None:
    stream = sys.stderr
  if width is None:
    try:
      # Try to read terminal width (ony for *nix systems)
      _,width = os.popen('stty size', 'r').read().split()
      width = int(width)
    except:
      width = 70
  tmpl = "%%-%ds" % max_key_len
  indent = ' ' * (max_key_len+pad)
  for k,v in data:
    print >>stream, tmpl % k,
    v = textwrap.fill(v, width=width,
        subsequent_indent=indent,
        initial_indent=indent)
    if len(k) <= max_key_len:
      # Use initial indent to account for printed flags, but remove it afterward
      print >>stream, " ", v[max_key_len+pad:]
    else:
      print >>stream, "\n%s" % v

def PrintModelParamHelp():
  params = MakeParams()
  traits = params.traits()
  # Display traits in alphabetical order.
  keys = sorted(n for n in params.trait_names() if not n.startswith('trait_'))
  # Format set of traits as a string.
  data = list()
  for k in keys:
    trait = traits[k]
    desc = trait.desc or ''
    if desc:
      idx = desc.index(' ')
      desc = desc[:idx].capitalize() + desc[idx:]
      desc = "%s. " % desc
    doc = (desc + "Must be %s. " % trait.full_info(params, '', '') +
        "Default is: %s" % pprint.pformat(trait.default))
    data.append(("%s:" % k, doc))
  PrintDict(data, max_key_len=25)

def Main(argv = None):
  options = MakeCliOptions()
  try:
    ParseCommandLine(options, argv = argv)
    if options.help.value:
      print >>sys.stderr, "Usage: [options]"
      PrintUsage(options)
      print >>sys.stderr
      print >>sys.stderr, "Model Parameters (and defaults):"
      PrintModelParamHelp()
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
