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

def main_iterable(options,xform_dir, image_dir,step_size,bbox_size):

  exp = LoadExperiment(os.path.join(xform_dir, "exp.dat"))
  model = exp.model
  image = Image.open(image_dir)
  image_copy = image.copy()
  # some variables to hold the origional size of the image for computation of scales later on
  orig_width, orig_height = image.size
  scale_layers = downsample(image_copy,100)

  print "Number of scales %d" % len(scale_layers)

  #process the image at each scale and grab features from c1 or c2 space
  results = list()
  for scale in scale_layers:

    # scale the image down given a scale tuple from scale_layers
    layer = []
    image_copy.resize(scale,Image.ANTIALIAS)
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
    if options == '-d':
        start_time = time.clock()


    for box in windows:
      # add the results of this layer to the results main list
      results.append(process_window(box,model,c1_layer,exp))

    if options == '-d':
        end_time = time.clock()
        taken = end_time - start_time
        print "Time Taken: ", (taken/60)

  crop = results[0]
  box = crop[2]
  box = map_to_image(exp,box)
  print box
  draw = ImageDraw.Draw(image)
  draw.rectangle(box,outline="red")
  print image.size
  image.show()

  #function to take an image and downsample it at some fixed amount given by command line argument at some point
def downsample(image_copy,scale_amount):
  layers = []
  height_ratio = .75

# down sample the image at specific intervals need to keep aspect ratio of 4:3
  for x in range(0,500,scale_amount):
    width,height = image_copy.size
    width = width - x
    height = width * .75
    size = (int(width),int(height))
    layers.append(size)
  return layers

# process the c1 crop given a bbox for the dimensions. Returns the svm result list
def process_window(bbox,model,c1_layer,exp):

    results = list()
    # get a 4d array and take chunks out of that array then flatten for svm
    w = c1_layer[:,:,bbox[0]:bbox[1],bbox[2]:bbox[3]]
    w_flat = w.flatten()

    # classify with trained svm passing in w_flat
    model = svm.ScaledSvm(classifier = exp.classifier, scaler = exp.scaler)

    #classify crop
    predicted_labels, acc, decision_values = model.Test(([w_flat], []))
    decision_values = decision_values[0]  # decision values contains a tuple of
                                          # per-instance measures, get the first
                                          # measure

    #store in results[i][b][bbox] which holds the decision d for a given classifier
    results.append(1)
    results.append(decision_values)
    results.append(bbox)
    r_array = np.array(results,dtype=object)
    return r_array

# map c1_layer bounding box back to image layer
def map_to_image(exp, bbox):
  """Given a glab experiment object 'exp' and a bounding box in C1 with
  upper-left coordinate (c1_x0, c1_y0) and lower-left coordinate (c1_x1,
  c1_y1).

  you can compute the corresponding bounding box in image
  coordinates as:"""
  c1_y0,c1_y1,c1_x0,c1_x1 = bbox
  from glimpse.models.viz2.layer_mapping import RegionMapper
  mapper = RegionMapper(exp.model.params)  # use the model parametersfrom the experiment
  c1_xrange = slice(c1_x0, c1_x1)
  print c1_xrange
  c1_yrange = slice(c1_y0, c1_y1)
  print c1_yrange
  img_xrange = mapper.MapC1ToImage(c1_xrange)
  img_yrange = mapper.MapC1ToImage(c1_yrange)
  img_x0, img_x1 = img_xrange.start, img_xrange.stop
  img_y0, img_y1 = img_yrange.start, img_yrange.stop
  """Thus, the bounding box in pixel space has upper-left coordinate
  (img_x0, img_y0) and lower-left coordinate (img_x1, img_y1)"""
  box = (img_x0,img_y0,img_x1,img_y1)
  return box

if __name__ == '__main__':
  if len(sys.argv) < 5:
    sys.exit("usage: %s DEBUG XFORM-DIR IMAGE WINDOW-STEP-SIZE" % sys.argv[0])
  options,xform_dir, image_dir, step_size = sys.argv[1:5]
  main_iterable(options,xform_dir, image_dir,int(step_size),24)
