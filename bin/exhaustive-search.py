#!/usr/bin/python

from glimpse.gditlib import TestSvm, ModelWrapper
from glimpse import util
import Image
import os
import svmutil
import sys

def GetCropBoundingBoxesFromImage(model, width, height):
  crop_width = 128
  output_width = 128
  step_size = 128
  for y in range(0, height - crop_width + 1, step_size):
    for x in range(0, width - crop_width + 1, step_size):
      box = (y, x, y + crop_width, x + crop_width)  # the bounding box
      yield box
  raise StopIteration

def main(xform_dir, image_path):
  image = Image.open(image_path).convert('L')
  model = ModelWrapper()
  bounding_boxes = list(GetCropBoundingBoxesFromImage(model, image.size[0],
      image.size[1]))
  assert len(bounding_boxes) > 0
  crops = [ model.ImageToState(image.crop(b)) for b in bounding_boxes ]
  c1_activity = model.ComputeC1Activity(crops)
  feature_transformer = util.Load(os.path.join(xform_dir,
      'feature-transformer'))
  features = map(feature_transformer.ComputeSvmFeatures, c1_activity)
  model = svmutil.svm_load_model(os.path.join(xform_dir, 'svm-model'))
  predicted_labels, decision_values = TestSvm(model, features, [])
  print "BOUNDING-BOX PREDICTED-LABEL CONFIDENCE"
  for box, label, value in zip(bounding_boxes, predicted_labels,
      decision_values):
    print box, label, value

if __name__ == '__main__':
  if len(sys.argv) < 3:
    sys.exit("usage: %s XFORM-DIR IMAGE" % sys.argv[0])
  xform_dir, image_path = sys.argv[1:3]
  main(xform_dir, image_path)
