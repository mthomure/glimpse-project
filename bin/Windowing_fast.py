#!/usr/bin/python

from glimpse.util import svm
from glimpse.glab import LoadExperiment
import glimpse.models
import glimpse.models.viz2
import Image
import ImageDraw
from itertools import imap
import numpy as np
import os
import sys
import time

def main_iterable(xform_dir, image_dir):

  exp = LoadExperiment(os.path.join(xform_dir, "exp.dat"))
  model = exp.model
  image = Image.open(image_dir)
  image.load
  image_copy = image.convert('L')

  scale_layers = downsample(image_copy,100)

  print "Number of scales %d" % len(scale_layers)

  #process the image at each scale and grab features from c1 or c2 space
  for scale in scale_layers:

    layer = []
    image_copy.thumbnail(scale,Image.ANTIALIAS)
    layer.append(image_copy)
    print layer

    # convert entire image over to c1 space
    image_layer = model.MakeStateFromImage(layer[0])
    s = model.BuildLayer(model.Layer.C1,image_layer,save_all = False)
    print "Layer processed"
    print len(s['c1'])
    # get a 4d array and take chunks out of that array then flatten




  #function to take an image and downsample it at some fixed amount given by command line argument
def downsample(image_copy,scale_amount):
  layers = []
  height_ratio = .75

# down sample the image at specific intervals need to keep aspect ratio of 4:3
  for x in range(0,1000,scale_amount):
    width,height = image_copy.size
    width = width - x
    height = width * .75
    size = (width,height)
    layers.append(size)
  return layers


if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit("usage: %s XFORM-DIR IMAGE" % sys.argv[0])
  xform_dir, image_dir= sys.argv[1:3]
  main_iterable(xform_dir, image_dir)
