#!/usr/bin/python

from glimpse.gditlib import GetDirContents, TestSvm, ModelWrapper
from glimpse import util
import os
import svmutil
import sys

def main(xform_dir, pos_image_dir, neg_image_dir):
  # Compute SVM feature vectors for positive images.
  pos_fnames = GetDirContents(pos_image_dir)
  model = ModelWrapper()
  pos_states = map(model.FilenameToState, pos_fnames)
  pos_c1_activity = model.ComputeC1Activity(pos_states)
  feature_transformer = util.Load(os.path.join(xform_dir,
      'feature-transformer'))
  pos_features = map(feature_transformer.ComputeSvmFeatures,
      pos_c1_activity)
  del pos_fnames, pos_c1_activity
  # Compute SVM feature vectors for negative images.
  neg_fnames = GetDirContents(neg_image_dir)
  neg_states = map(model.FilenameToState, neg_fnames)
  neg_c1_activity = model.ComputeC1Activity(neg_states)
  neg_features = map(feature_transformer.ComputeSvmFeatures,
      neg_c1_activity)
  del neg_fnames, neg_c1_activity
  # Load and apply the SVM classifier to feature vectors.
  model = svmutil.svm_load_model(os.path.join(xform_dir, 'svm-model'))
  predicted_labels, decision_values = TestSvm(model, pos_features,
      neg_features)
  for pl, dv in zip(predicted_labels, decision_values):
    print pl, dv

if __name__ == '__main__':
  if len(sys.argv) < 4:
    sys.exit("usage: %s XFORM-DIR POS-DIR NEG-DIR" % sys.argv[0])
  xform_dir, pos_image_dir, neg_image_dir = sys.argv[1:4]
  main(xform_dir, pos_image_dir, neg_image_dir)
