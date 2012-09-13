"""Provides a command-line interface for the GLAB module."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import os

from glimpse import pools
from glimpse import util
from .misc import *

def _InitCli(pool_type = None, cluster_config = None, model_name = None,
    params = None, edit_params = False, layer = None, verbose = 0, **opts):
  if verbose > 0:
    Verbose(True)
    if verbose > 1:
      logging.getLogger().setLevel(logging.INFO)
  # Make the worker pool
  if pool_type != None:
    pool_type = pool_type.lower()
    if pool_type in ('c', 'cluster'):
      UseCluster(cluster_config)
    elif pool_type in ('m', 'multicore'):
      pool = pools.MulticorePool()
      SetPool(pool)
    elif pool_type in ('s', 'singlecore'):
      pool = pools.SinglecorePool()
      SetPool(pool)
    else:
      raise util.UsageException("Unknown pool type: %s" % pool_type)
  try:
    SetModelClass(model_name)
  except ValueError:
    raise util.UsageException("Unknown model (-m): %s" % model_name)
  SetParams(params)
  SetLayer(layer)
  if edit_params:
    GetParams().configure_traits()
  # At this point, all parameters needed to create Experiment object are set.

def _FormatCliResults(svm_decision_values = False, svm_predicted_labels = False,
    **opts):
  exp = GetExperiment()
  if exp.train_results != None:
    print "Train Accuracy: %.3f" % exp.train_results['accuracy']
  if exp.test_results != None:
    print "Test Accuracy: %.3f" % exp.test_results['accuracy']
    test_images = exp.test_images
    test_results = exp.test_results
    if svm_decision_values:
      if 'decision_values' not in test_results:
        logging.warn("Decision values are unavailable.")
      else:
        decision_values = test_results['decision_values']
        print "Decision Values:"
        for cls in range(len(test_images)):
          print "\n".join("%s %s" % _
              for _ in zip(test_images[cls], decision_values[cls]))
    if svm_predicted_labels:
      if 'predicted_labels' not in test_results:
        logging.warn("Predicted labels are unavailable.")
      else:
        predicted_labels = test_results['predicted_labels']
        print "Predicted Labels:"
        for cls in range(len(test_images)):
          print "\n".join("%s %s" % _
              for _ in zip(test_images[cls], predicted_labels[cls]))
  else:
    print "No results available."

def _RunCli(prototypes = None, prototype_algorithm = None, num_prototypes = 10,
    corpus = None, use_svm = False, compute_features = False, raw = False,
    result_path = None, cross_validate = False, verbose = 0, balance = False,
    corpus_subdirs = None, compute_raw_features = False, **opts):
  if corpus != None:
    SetCorpus(corpus, balance = balance)
  elif corpus_subdirs:  # must be not None and not empty list
    SetCorpusSubdirs(corpus_subdirs, balance = balance)
  num_prototypes = int(num_prototypes)
  if prototypes != None:
    SetS2Prototypes(prototypes)
  if prototype_algorithm != None:
    prototype_algorithm = prototype_algorithm.lower()
    if prototype_algorithm == 'imprint':
      ImprintS2Prototypes(num_prototypes)
    elif prototype_algorithm == 'uniform':
      MakeUniformRandomS2Prototypes(num_prototypes)
    elif prototype_algorithm == 'shuffle':
      MakeShuffledRandomS2Prototypes(num_prototypes)
    elif prototype_algorithm == 'histogram':
      MakeHistogramRandomS2Prototypes(num_prototypes)
    elif prototype_algorithm == 'normal':
      MakeNormalRandomS2Prototypes(num_prototypes)
    else:
      raise util.UsageException("Invalid prototype algorithm "
          "(%s), expected 'imprint' or 'random'." % prototype_algorithm)
  if compute_features or compute_raw_features:
    ComputeFeatures(raw = compute_raw_features)
  if use_svm:
    RunSvm(cross_validate)
    if verbose > 0:
      _FormatCliResults(**opts)
  if result_path != None:
    StoreExperiment(result_path)

def CommandLineInterface(**opts):
  """Entry point for command-line interface handling."""
  _InitCli(**opts)
  _RunCli(**opts)

def main():
  try:
    opts = dict()
    opts['verbose'] = 0
    opts['corpus_subdirs'] = []
    cli_opts, _ = util.GetOptions('bc:C:el:m:n:o:p:P:r:st:vx',
        ['balance', 'corpus=', 'corpus-subdir=', 'cluster-config=',
        'compute-features', 'compute-raw-features', 'edit-options', 
        'layer=', 'model=',
        'num-prototypes=', 'options=', 'prototype-algorithm=', 'prototypes=',
        'results=', 'svm', 'svm-decision-values',
        'svm-predicted-labels', 'pool-type=', 'verbose', 'cross-validate'])
    for opt, arg in cli_opts:
      if opt in ('-b', '--balance'):
        opts['balance'] = True
      elif opt in ('-c', '--corpus'):
        opts['corpus'] = arg
      elif opt in ('-C', '--corpus-subdir'):
        opts['corpus_subdirs'].append(arg)
      elif opt == '--cluster-config':
        # Use a cluster of worker nodes
        opts['cluster_config'] = arg
      elif opt == '--compute-features':
        opts['compute_features'] = True
      elif opt == '--compute-raw-features':
        opts['compute_raw_features'] = True
      elif opt in ('-e', '--edit-options'):
        opts['edit_params'] = True
      elif opt in ('-l', '--layer'):
        opts['layer'] = arg
      elif opt in ('-m', '--model'):
        opts['model_name'] = arg
      elif opt in ('-n', '--num-prototypes'):
        opts['num_prototypes'] = int(arg)
      elif opt in ('-o', '--options'):
        opts['params'] = util.Load(arg)
      elif opt in ('-p', '--prototype-algorithm'):
        opts['prototype_algorithm'] = arg.lower()
      elif opt in ('-P', '--prototypes'):
        opts['prototypes'] = util.Load(arg)
      elif opt in ('-r', '--results'):
        opts['result_path'] = arg
      elif opt in ('-s', '--svm'):
        opts['use_svm'] = True
      elif opt == '--svm-decision-values':
        opts['svm_decision_values'] = True
        opts['verbose'] = max(1, opts['verbose'])
        opts['svm'] = True
      elif opt == '--svm-predicted-labels':
        opts['svm_predicted_labels'] = True
        opts['verbose'] = max(1, opts['verbose'])
        opts['svm'] = True
      elif opt in ('-t', '--pool-type'):
        opts['pool_type'] = arg.lower()
      elif opt in ('-v', '--verbose'):
        opts['verbose'] += 1
      elif opt in ('-x', '--cross-validate'):
        opts['cross_validate'] = True
    CommandLineInterface(**opts)
  except util.UsageException, ex:
    util.Usage("[options]\n"
        "  -b, --balance                   Choose equal number of images per "
        "class\n"
        "  -c, --corpus=DIR                Use corpus directory DIR\n"
        "  -C, --corpus-subdir=DIR         Specify subdirectories (using -C"
        " repeatedly)\n"
        "                                  instead of single corpus directory"
        " (with -c)\n"
        "      --cluster-config=FILE       Read cluster configuration from "
        "FILE\n"
        "      --compute-features          Compute feature vectors (implied "
        "by -s)\n"
        "      --compute-raw-features      (advanced) Compute raw layer features "
        "(i.e.,\n"
        "                                  do not flatten feature vectors)\n"
        "  -e, --edit-options              Edit model options with a GUI\n"
        "  -l, --layer=LAYR                Compute feature vectors from LAYR "
        "activity\n"
        "  -m, --model=MODL                Use model named MODL\n"
        "  -n, --num-prototypes=NUM        Generate NUM S2 prototypes\n"
        "  -o, --options=FILE              Read model options from FILE\n"
        "  -p, --prototype-algorithm=ALG   Generate S2 prototypes according "
        "to algorithm\n"
        "                                  ALG (one of 'imprint', 'uniform', "
        "'shuffle',\n"
        "                                  'histogram', or 'normal')\n"
        "  -P, --prototypes=FILE           Read S2 prototypes from FILE "
        "(overrides -p)\n"
        "  -r, --results=FILE              Store results to FILE\n"
        "  -s, --svm                       Train and test an SVM classifier\n"
        "      --svm-decision-values       Print the pre-thresholded SVM "
        "decision values\n"
        "                                  for each test image (implies -vs)\n"
        "      --svm-predicted-labels      Print the predicted labels for each "
        "test image\n"
        "                                  (implies -vs)\n"
        "  -t, --pool-type=TYPE            Set the worker pool type (one of "
        "'multicore',\n"
        "                                  'singlecore', or 'cluster')\n"
        "  -v, --verbose                   Enable verbose logging\n"
        "  -x, --cross-validate            Compute test accuracy via cross-"
        "validation\n"
        "                                  instead of fixed training/testing "
        "split",
        ex
    )

if __name__ == '__main__':
  main()

