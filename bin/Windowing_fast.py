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

def main_iterable(xform_dir, image_dir,step_size, bbox_size):

  exp = LoadExperiment(os.path.join(xform_dir, "exp.dat"))
  model = exp.model
  image = Image.open(image_dir)
  image.load
  image_copy = image.convert('L')

  scale_layers = downsample(image_copy,100)

  print "Number of scales %d" % len(scale_layers)

  #process the image at each scale and grab features from c1 or c2 space
  results = list()
  for scale in scale_layers:

    # downsample the image
    layer = []
    image_copy.thumbnail(scale,Image.ANTIALIAS)
    layer.append(image_copy)

    # convert entire image over to c1 space
    image_layer = model.MakeStateFromImage(layer[0])
    s = model.BuildLayer(model.Layer.C1,image_layer,save_all = False)
    print "Layer processed"
    c1_layer = np.array(s['c1'])

    # holds the sliding windows for the given layer
    windows = list()

    # create bounding boxes for the given scale layer over all positions in the c1 map
    for i in range(0,c1_layer.shape[3], step_size):
      for j in range(0,c1_layer.shape[2], step_size):

        y0 = 0 + i
        y1 = bbox_size + i
        x0= 0 + j
        x1 =  bbox_size + j

        if y1 >= c1_layer.shape[2]:
          break

        if x1 >= c1_layer.shape[3]:
          break

        bbox = (y0,y1,x0,x1)
        windows.append(bbox)
        # call process window
    for box in windows:
      print process_window(box,model,c1_layer,exp)

    print len(windows)
    #x length
    print c1_layer.shape[3]
    #y length
    print c1_layer.shape[2]
    break


  #function to take an image and downsample it at some fixed amount given by command line argument at some point
def downsample(image_copy,scale_amount):
  layers = []
  height_ratio = .75

# down sample the image at specific intervals need to keep aspect ratio of 4:3
  for x in range(0,500,scale_amount):
    width,height = image_copy.size
    width = width - x
    height = width * .75
    size = (width,height)
    layers.append(size)
  return layers

# process the c1 crop given a bbox for the dimensions. Returns the svm result list
def process_window(bbox,model,c1_layer,exp):
    results = np.zeros((1,2))
    print results
    # get a 4d array and take chunks out of that array then flatten for svm
    w = c1_layer[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]]
    w_flat = w.flatten()
    print w_flat.shape

    # classify with trained svm passing in w_flat
    model = svm.ScaledSvm(classifier = exp.classifier, scaler = exp.scaler)


    #classify crop
    predicted_labels, acc, decision_values = model.Test(([w_flat], []))
    decision_values = decision_values[0]  # decision values contains a tuple of
                                          # per-instance measures, get the first
                                          # measure

    #store in results[i][b] which holds the decision d for a given classifier and return
    results[0][0] = 1
    results[0][1] = decision_values
    return results

if __name__ == '__main__':
  if len(sys.argv) < 5:
    sys.exit("usage: %s XFORM-DIR IMAGE WINDOW-STEP-SIZE BBOX-SIZE" % sys.argv[0])
  xform_dir, image_dir, step_size, bbox_size = sys.argv[1:5]
  main_iterable(xform_dir, image_dir,int(step_size),int(bbox_size))
