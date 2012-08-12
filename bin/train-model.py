# Called by train-models.sh. See that script for usage.

import os
import random
import sys

from glimpse import glab
from glimpse.glab.experiment import DirReader
from glimpse.models import ml
from glimpse import util

def main(model_path, model_name, train_pos_path, train_neg_path, params_path, num_prototypes):
  # Configure the model, loading parameters from disk.
  glab.SetModelClass(ml.Model)
  glab.SetParams(util.Load(params_path))
  # Read training images from disk.
  reader = DirReader()
  train_images = map(reader.ReadFiles, (train_pos_path, train_neg_path))
  num_images = min(map(len, train_images))
  # Randomly permute order of images (in-place).
  map(random.shuffle, train_images)
  # Use balanced number of positive/negative instances.
  train_images = [ imgs[:num_images] for imgs in train_images ]
  test_images = ([], [])  # empty test set with two classes
  # Set the class names to the training set directories.
  # XXX assumes classes have different directory names.
  classes = map(os.path.basename, (train_pos_path, train_neg_path))
  exp = glab.GetExperiment()
  exp.SetTrainTestSplit(train_images, test_images, classes)
  print "Imprinting S2 Prototypes"
  exp.ImprintS2Prototypes(int(num_prototypes))
  print "Transforming Images and Training Classifier"
  exp.TrainSvm()
  # Store trained model to disk.
  exp_path = os.path.join(model_path, model_name + ".dat")
  util.Store(exp, exp_path)

if __name__ == '__main__':
  if len(sys.argv) < 7:
    sys.exit("usage: %s MODEL_PATH MODEL POS_TRAIN_DIR NEG_TRAIN_DIR PARAMS NUM_PROTOTYPES" % sys.argv[0])
  main(*sys.argv[1:7])
