#!/usr/bin/python

from glimpse.glab import LoadExperiment
from glimpse.models.viz2 import Model as Viz2Model
from glimpse.models.viz2.layer_mapping import RegionMapper
from glimpse.util import svm
import Image
import ImageDraw
import logging
import numpy as np
import os
import sys
import time

def MakeBoundingBoxes(height, width, step_size, box_size):
  """Create bounding boxes in the X-Y plane.
  height, width -- (int) size of the plane
  step_size -- (int) distance between top-left corner of adjacent boxes
  box_size -- (int) width and height of each box
  RETURN (2D list) chosen bounding boxes in the format (y0, y1, x0, x1), where
  (y0, x0) gives the upper-left corner of the box (i.e., y0 and x0 are
  inclusive), and (y1, x1) gives the unit just beyond the lower-right corner
  (i.e., y1 and x1 are exclusive).
  """
  # Holds the sliding windows for the given layer
  windows = list()
  # Create bounding boxes for the given scale layer over all positions in the c1
  # map
  for j in range(0, height - box_size, step_size):
    y0, y1 = j, j + box_size
    for i in range(0, width - box_size, step_size):
      x0, x1 = i, i + box_size
      bbox = np.array([y0, y1, x0, x1], np.int)
      windows.append(bbox)
  return windows

def ScaleImage(image, scale):
  """Scale an image (either up-sampling or down-sampling) by a given ratio."""
  width, height = image.size
  width = int(width * scale)
  height = int(height * scale)
  return image.resize((width, height), Image.ANTIALIAS)

class Windower(object):

  def __init__(self, glimpse_model, svm_model, step_size, bbox_size,
      debug = False):
    assert isinstance(glimpse_model, Viz2Model), "Wrong model type"
    self.glimpse_model = glimpse_model
    self.svm_model = svm_model
    self.step_size = step_size
    self.bbox_size = bbox_size
    self.debug = debug
    self.mapper = RegionMapper(glimpse_model.params)

  def ChooseImageScales(self, image_size):
    """Choose the scaling (down-sampling) ratios for a given image. This is a member function so
    that the scaling algorithm can be easily overridden."""
    image_width = image_size[0]
    # Choose image scales by requesting a fixed image size, as a offset from the
    # current image width.
    new_widths = image_width - np.arange(0, 1000, 100)
    return new_widths / float(image_width)

  def MapC1RegionToImageBox(self, bbox, scale):
    """ Map C1 layer coordinates to the corresponding coordinates in image
    space, and adjusts for image scaling.
    bbox -- (1D array-like) C1 region in the format of (y0, y1, x0, x1)
    scale -- (float) scaling ratio between scaled image size and original input
             image size
    RETURN (1D np.array) bounding box in original input image coordinates.
    """
    c1_y0, c1_y1, c1_x0, c1_x1 = bbox
    c1_yrange = slice(c1_y0, c1_y1)
    c1_xrange = slice(c1_x0, c1_x1)
    img_yrange = self.mapper.MapC1ToImage(c1_yrange)
    img_xrange = self.mapper.MapC1ToImage(c1_xrange)
    img_y0, img_y1 = img_yrange.start, img_yrange.stop
    img_x0, img_x1 = img_xrange.start, img_xrange.stop
    return (np.array([img_x0, img_y0, img_x1, img_y1]) / scale).astype(np.int)

  def ClassifyC1Window(self, crop):
    """Compute SVM output for a single C1 crop.
    crop -- (3D np.array) region of C1 activity
    RETURN predicted label and decision value
    """
    crop = crop.flatten()
    pos_instances = [crop]
    neg_instances = []
    all_instances = pos_instances, neg_instances
    predicted_labels, _, dvalues = self.svm_model.Test(all_instances)
    return predicted_labels[0], dvalues[0][0]

  def ProcessScale(self, scaled_image):
    logging.info("ProcessScale() -- scaled image size: %s" % \
        (scaled_image.size,))
    # Compute C1 layer activity.
    image_layer = self.glimpse_model.MakeStateFromImage(scaled_image)
    output_layer = self.glimpse_model.Layer.C1
    output_state = self.glimpse_model.BuildLayer(output_layer, image_layer,
        save_all = False)
    c1_layer = np.array(output_state[output_layer.id])
    logging.info("C1 layer shape for scale: %s" % (c1_layer.shape,))
    # Find and process all bounding boxes.
    c1_height, c1_width = c1_layer.shape[2:]
    bboxes = MakeBoundingBoxes(c1_height, c1_width, self.step_size,
        self.bbox_size)
    dvalues_per_bbox = list()
    start_time = time.time()
    for bbox in bboxes:
      # get a 4d array and take chunks out of that array then flatten for svm
      window = c1_layer[ :, :, bbox[0] : bbox[1], bbox[2] : bbox[3] ]
      predicted_label, dvalue = self.ClassifyC1Window(window)
      # Return the pre-thresholded decision value for this bounding box
      dvalues_per_bbox.append(dvalue)
    end_time = time.time()
    logging.info("Time to process scale (%d boxes): %.2f secs" % (len(bboxes),
        end_time - start_time))
    return bboxes, np.array(dvalues_per_bbox)

  def Process(self, image):
    """Choose and classify bounding boxes for the given image.
    image -- (Image) input image
    RETURN (float list) scaling ratios, (3D list) bounding box for each region
           in scaled C1 coordinates, (list of np.ndarray) decision value for
           each region
    """
    image_scales = self.ChooseImageScales(image.size)
    logging.info("Image scales: %s", (image_scales,))
    bboxes_per_scale = list()
    dvalues_per_scale = list()
    results = [ self.ProcessScale(ScaleImage(image, scale))
        for scale in image_scales ]
    bboxes_per_scale, dvalues_per_scale = zip(*results)
    logging.info("There are %d results" % sum(map(len, dvalues_per_scale)))
    return image_scales, bboxes_per_scale, dvalues_per_scale

def main(xform_dir, image_path, step_size, bbox_size, threshold, debug = False):
  if debug:
    logging.getLogger().setLevel(logging.INFO)
  exp = LoadExperiment(os.path.join(xform_dir, "exp.dat"))
  image = Image.open(image_path)
  svm_model = svm.ScaledSvm(classifier = exp.classifier, scaler = exp.scaler)
  windower = Windower(exp.model, svm_model, step_size, bbox_size, debug = debug)
  image_scales, bboxes_per_scale, dvalues_per_scale = windower.Process(image)

  """
  given a bounding box and scale , draw the box to the image
  """
  def ShowBoundingBox(bbox, scale, title = ""):

    image_copy = image.copy()

    bbox = windower.MapC1RegionToImageBox(bbox, scale)
    print "Bounding Box[%s]: %s" % (title, bbox)
    draw = ImageDraw.Draw(image_copy)
    draw.rectangle(tuple(bbox), outline = "red")
    #image_copy.save('bbox - %s.png' % title)
    #image_copy.show()
    return image_copy

  """ suppress regions of the image using Local Neighborhood suppresion. delta is
      a gaussian value that is small and based on the scale of the maximum crop,
      or more easily just some parameter in the algorithm.
  """
  def suppress_neighborhood(neighborhood,bboxes_per_scale,scale,gauss = False):
    print neighborhood
    pass

  # As a test, draw the bounding box for the second crop.
  scale = 1
  bbox = bboxes_per_scale[scale][1]
  scale = image_scales[scale]
  print image_scales
  #ShowBoundingBox(bbox, scale, "arbitrary")
  # Alternatively, we could search the image for the best match.
  max_scale = np.argmax(dvalues.max() for dvalues in dvalues_per_scale)
  max_region = dvalues_per_scale[max_scale].argmax()
  bbox = bboxes_per_scale[max_scale][max_region]
  scale = image_scales[max_scale]
  #ShowBoundingBox(bbox, scale, "best-match")

  #Or try to look for areas based on some threshold amount over all of the scales
  bboxes = list()
  for scale_index in range(0,len(dvalues_per_scale)):
    for class_val in range(0,len(dvalues_per_scale[scale_index])):
      if dvalues_per_scale[scale_index][class_val] > threshold:
        image = ShowBoundingBox(bboxes_per_scale[scale_index][class_val],image_scales[scale_index],"Threshold match")
        suppress_neighborhood(bboxes_per_scale[scale_index][class_val],bboxes_per_scale,image_scales[scale])

  image.save('../../image_5.jpg')


if __name__ == '__main__':
  if len(sys.argv) < 5:
    sys.exit("usage: %s XFORM-DIR IMAGE WINDOW-STEP-SIZE THRESHOLD" % sys.argv[0])
  xform_dir, image_path, step_size,threshold = sys.argv[1:5]
  # Use a bounding box size (in C1 coordinates) of 24 units, which is equivalent
  # to 128 pixels in image space.
  bbox_size = 24
  main(xform_dir, image_path, int(step_size), bbox_size,float(threshold), debug = True)
