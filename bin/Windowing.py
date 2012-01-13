#!/usr/bin/python

from glimpse.util import svm
from glimpse.glab import LoadExperiment
import Image
import ImageDraw
from itertools import imap
import numpy as np
import os
import sys
import time

def main_iterable(options,xform_dir, image_dir, spacing):
  exp = LoadExperiment(os.path.join(xform_dir, "exp.dat"))
  model = exp.model
  image = Image.open(image_dir)
  image.load
  image_copy = image.convert('L')



  # build amap of confidence values over position and scale
  image_confidence_map = []

  #downsample the image keeping 4:3 aspect ratio till can no longer take crops
  boxes = subwindows(image_copy, ystep = spacing, xstep = spacing, box_size = 128)
  image_confidence_map.append(boxes)
  print "crops done (%d boxes)" % len(boxes)
  # Compute SVM feature vectors for image crops

  def crop_image(box):
    crop = image_copy.crop(box)

    if crop.size != (128,128):
      size = (128,128)
      crop = crop.resize(size)
    return crop

  crops = imap(crop_image, boxes)

  img_states = imap(model.MakeStateFromImage, crops)
  print "states generated"

  if options == '-d':
    start_time_c1 = time.clock()

  img_c1_activity = exp.ComputeFeaturesFromInputStates(img_states,
      as_iterator = True)

  if options == '-d':
    current_time = time.clock()
    time_taken = current_time - start_time_c1
    print "Time compute image features %d" % time_taken

  def progress_updater(xs, update_every = 10):
    count = 0
    for x in xs:
      count += 1
      if count >= update_every:
        print ".",
        sys.stdout.flush()
        count = 0
      yield x

  img_c1_activity = progress_updater(img_c1_activity)
  # Apply the SVM classifier to feature vectors.
  model = svm.ScaledSvm(classifier = exp.classifier, scaler = exp.scaler)
  print "FILE PREDICTED-LABEL CONFIDENCE"
  per_crop_decision_values = list()

  if options == '-d':
     start_time = time.clock()
     crops_per_min = 0

  for c1_features in img_c1_activity:

    if options == '-d':
      current = time.clock()
      if current - start_time >= 1.0:
        print crops_per_min
        print (current - start_time)
        break
      crops_per_min = crops_per_min + 1

    predicted_labels, acc, decision_values = model.Test(([c1_features], []))
    decision_values = decision_values[0]  # decision values contains a tuple of
                                          # per-instance measures, get the first
                                          # measure
    #print predicted_labels[0], decision_values[0]
    per_crop_decision_values.append(decision_values[0])

  print_max(image, boxes, per_crop_decision_values)


  #~ # Evaluate the iterable
  #~ img_c1_activity = list(img_c1_activity)
  #~ if len(img_c1_activity) == 0:
    #~ return
  #~ predicted_labels, acc, decision_values = model.Test((img_c1_activity, []))
  #~ decision_values = [ x[0] for x in decision_values ]
#~
  #~ print "FILE PREDICTED-LABEL CONFIDENCE"
  #~ for pl, dv in zip(predicted_labels, decision_values):
    #~ print pl, dv
  #~ print_max(image, boxes, decision_values)

#~ # create a list of crops to process by tiling the windows
#~ def create_croplist(image,window_size):
    #~ # take an image return a list of crops of that image
    #~ width,height = image.size
    #~ croplist = []
#~
    #~ # perform the crops and add them to the croplist
    #~ for y in range(0,height - window_size, window_size):
      #~ for x in range(0,width - window_size, window_size):
          #~ box = (x,y,x+window_size,y+window_size)
          #~ crop = image.crop(box)
          #~ size = (128,128)
          #~ crop = crop.resize(size)
          #~ croplXist.append(crop)
    #~ return croplist

#Calculate all the subwindow crops for the image
def subwindows(image, ystep, xstep, box_size):
  width, height = image.size
  boxes = []
  for y in range(0, height, ystep):
    for x in range(0, width , xstep):

      box = (x, y, x + box_size, y + box_size)
      boxes.append(box)
  return boxes


#~ def subwindow_crops(image):
  #~ width,height = image.size
  #~ crops = []
  #~ for y in range(0,height-256,10):
    #~ for x in range(0,width-256,10):
      #~ box = (x,y,x+256,y+256)
      #~ crop = image.crop(box)
      #~ size = (256,256)
      #~ crop = crop.resize(size)
      #~ crops.append(crop)
  #~ return crops


def print_max(image,boxes,results):
  assert(len(boxes) == len(results))
  draw = ImageDraw.Draw(image)
  for x, box in zip(results, boxes):
    if float(x) > 0.1:
      draw.rectangle(box, outline="green")

  image.show()

  #calculate percentage of height and width to keep aspect ratio
  #copy image and downsample

# determine the global maximum neighborhoods of an image by taking current max bounding box and loop till some threshhold

if __name__ == '__main__':
  if len(sys.argv) < 4:
    sys.exit("usage: %s Debug XFORM-DIR IMAGE WINDOW-STEP-SIZE " % sys.argv[0])
  options,xform_dir, image_dir, spacing = sys.argv[1:5]
  main_iterable(options,xform_dir, image_dir, int(spacing))
