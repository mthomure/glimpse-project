#!/usr/bin/python

from glimpse.glab import *
import os
import sys

def GetDirContents(dir_path):
  return [ os.path.join(dir_path, f) for f in os.listdir(dir_path) ]

def main(xform_dir, pos_image_dir, neg_image_dir):
 
  SetExperiment(layer = 'c1')
  pos_images = GetDirContents(pos_image_dir)
  neg_images = GetDirContents(neg_image_dir)
  classes = map(os.path.basename, (pos_image_dir, neg_image_dir))
  train_split = pos_images, neg_images
  test_split = ([], [])  # no test data
  SetTrainTestSplit(train_split, test_split, classes)
  RunSvm()
  StoreExperiment(os.path.join(xform_dir, 'exp.dat'))

if __name__ == '__main__':
  if len(sys.argv) < 4:
    sys.exit("usage: %s XFORM-DIR POS-DIR NEG-DIR" % sys.argv[0])
  xform_dir, pos_image_dir, neg_image_dir = sys.argv[1:4]
  main(xform_dir, pos_image_dir, neg_image_dir)
