#!/usr/bin/python

from glimpse.gditlib import TestSvm, ModelWrapper
from glimpse import util
import Image
import os
import svmutil
import sys

def GetCropsFromImage(model, image):
  height, width = image.size
  crop_width = 128
  output_width = 128
  step_size = 90
  for y in range(0, height - crop_width, step_size):
    for x in range(0, width - crop_width, step_size):
      box = (y, x, y + crop_width, x + crop_width)  # the bounding box
      crop = image.crop(box)
      yield model.ImageToState(crop)
  raise StopIteration

def main(xform_dir, image_path):
  image = Image.open(image_path).convert('L')
  model = ModelWrapper()
  crops = list(GetCropsFromImage(model, image))
  assert len(crops) > 0
  c1_activity = model.ComputeC1Activity(crops)
  feature_transformer = util.Load(os.path.join(xform_dir,
      'feature-transformer'))
  features = map(feature_transformer.ComputeSvmFeatures, c1_activity)
  model = svmutil.svm_load_model(os.path.join(xform_dir, 'svm-model'))
  predicted_labels, decision_values = TestSvm(model, features, [])
  print predicted_labels, decision_values

if __name__ == '__main__':
  if len(sys.argv) < 3:
    sys.exit("usage: %s XFORM-DIR IMAGE" % sys.argv[0])
  xform_dir, image_path = sys.argv[1:3]
  main(xform_dir, image_path)
