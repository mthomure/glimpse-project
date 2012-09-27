# Called by apply-model.sh. See that script for usage.
# Author: Mick Thomure
# Date: 8/12/2012

from glimpse import util  # must load glimpse.util first to avoid PIL "hash collision"

import csv
import numpy as np
import os
from scipy.io import loadmat
import sys

def MakeInputState(model, image_path):
  """Read image data from disk, and return a corresponding input state."""
  if image_path.endswith(".mat"):
    # Read pixel data from "image" entry of matlab dictionary.
    matlab_data = loadmat(image_path)
    keys = set(matlab_data.keys()).intersection(("IMAGE", "image", "img", "im", "data"))
    if len(keys) == 0:
      sys.exit("No image data found in matlab file.")
    image = matlab_data[list(keys)[0]]
    return model.MakeStateFromImage(image)
  # Read pixel data from image file.
  return model.MakeStateFromFilename(image_path)

def main(models_path, model_name, *image_paths):
  # Load the model information from disk.
  exp_path = os.path.join(models_path, model_name + ".dat")
  if not os.path.exists(exp_path):
    sys.exit("Model not found: %s" % model_name)
  exp = util.Load(exp_path)
  # Load image data.
  images = [ MakeInputState(exp.model, path) for path in image_paths ]
  # Apply the model to compute features.
  ftrs = np.array(exp.GetStateFeatures(images))
  # Get SVM decision value for each image. Predicted label is sign of decision value.
  decision_values = exp.classifier.decision_function(ftrs)
  # Write decision values as text, with one line per input image.
  writer = csv.writer(sys.stdout)
  writer.writerows(decision_values)

if __name__ == '__main__':
  if len(sys.argv) < 3:
    sys.exit("usage: %s MODELS_PATH MODEL IMAGE [IMAGE ...]" % sys.argv[0])
  main(*sys.argv[1:])
