#!/usr/bin/python

from glimpse.glab import LoadExperiment
from glimpse.util import svm
import os
import sys

def GetDirContents(dir_path):
  return [ os.path.join(dir_path, f) for f in os.listdir(dir_path) ]

def main(xform_dir, pos_image_dir, neg_image_dir):
  exp = LoadExperiment(os.path.join(xform_dir, "exp.dat"))
  pos_fnames = GetDirContents(pos_image_dir)
  print pos_fnames
  pos_features = exp.ComputeFeaturesFromInputStates(map( exp.model.MakeStateFromFilename, pos_fnames))
  neg_fnames = GetDirContents(neg_image_dir)
  neg_features = exp.ComputeFeaturesFromInputStates(map(
      exp.model.MakeStateFromFilename, neg_fnames))
  model = svm.ScaledSvm(classifier = exp.classifier, scaler = exp.scaler)
  predicted_labels, acc, decision_values = model.Test((pos_features,
      neg_features))

  print "FILE PREDICTED-LABEL CONFIDENCE"
  for f, pl, dv in zip(pos_fnames + neg_fnames, predicted_labels,
      decision_values):
    print f, pl, dv

if __name__ == '__main__':
  if len(sys.argv) < 4:
    sys.exit("usage: %s XFORM-DIR POS-DIR NEG-DIR" % sys.argv[0])
  xform_dir, pos_image_dir, neg_image_dir = sys.argv[1:4]
  main(xform_dir, pos_image_dir, neg_image_dir)
