#!/usr/bin/env python 

import cPickle as pickle
from sklearn.datasets import dump_svmlight_file
import sys

from glimpse.experiment import *

def Main(input_file, train_file, test_file):
  with open(input_file) as fh:
    exp = pickle.load(fh)
  ftrs = ExtractFeatures(Layer.C2, exp.extractor.activation)
  trng = GetTrainingSet(exp)
  dump_svmlight_file(ftrs[trng], exp.corpus.labels[trng] + 1, train_file, zero_based=False)
  dump_svmlight_file(ftrs[~trng], exp.corpus.labels[~trng] + 1, test_file, zero_based=False)
  print "Categories"
  print "----------"
  print "\n".join("%d - %s" % (index+1,name) for (index,name) in enumerate(exp.corpus.class_names))

if __name__ == '__main__':
  if len(sys.argv) < 4:
    sys.exit("usage: %s EXP.pkl TRAIN.svm TEST.svm" % sys.argv[0])
  Main(*sys.argv[1:4])
