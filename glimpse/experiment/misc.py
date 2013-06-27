# Miscellaneous bits of experimental code.

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

# TODO: we should probably move ExtractHistogramFeatures() to .utils, and rename
# .misc to .analysis

import math
import numpy as np
from scipy.misc import toimage, fromimage
import Image

from glimpse.models.ml import Layer
from glimpse.models.base import BuildLayer
from glimpse.models.ml.misc import GetS2ReceptiveField
from glimpse.util.garray import FlattenArrays
from glimpse.util.gimage import MakeScalePyramid
from glimpse.util.gplot import Show2dArray, Show3dArray

from .experiment import ExpError, ResolveLayers

def ExtractHistogramFeatures(layers, states, low=0, high=0.2, bins=None):
  """Compute image features as a histogram over location and scale.

  :type layers: str or list of str
  :param layers: One or more model layers to compute.
  :type states: list of :class:`BaseState`
  :param states: Model state for each image.
  :param float low: Minimum range for histogram.
  :param float high: Maximum range for histogram.
  :param float bins: Number of bins in histogram.
  :rtype: 2-d array of float
  :returns: Feature vectors.

  """
  layers = ResolveLayers(layers)
  if len(layers) < 1:
    raise ValueError("Must specify one or more layers from which to build "
        "features")
  results = list()
  for st in states:
    r = list()
    for layer in layers:
      # organize by feature band, and concatenate across scales
      xs = map(FlattenArrays, zip(*st[layer]))
      ys = [ np.histogram(x, range=(0,.2), bins=bins)[0] for x in xs ]
      r.append(ys)
    results.append(np.array(r).flatten())
  return np.array(results)

def GetImagePaths(exp):
  """Returns the filename for each image in the corpus."""
  return exp.corpus.paths

def GetLabelNames(exp):
  """Returns the class name for each image in the corpus."""
  return exp.corpus.class_names[exp.corpus.labels]

def GetParams(exp):
  """Returns the model parameters for the experiment."""
  model = exp.extractor.model
  if model is None:
    raise ExpError("Experiment has no model")
  return model.params

def GetNumPrototypes(exp, kwidth=0):
  """Return the number of S2 prototypes in the model.

  :param int kwidth: Index of kernel shape.

  """
  return len(exp.extractor.model.s2_kernels[kwidth])

def GetImprintLocation(exp, prototype=0, kwidth=0):
  """Return the image location from which a prototype was imprinted.

  This requires that the prototypes were learned by imprinting.

  :param int prototype: Index of S2 prototype.
  :param int kwidth: Index of kernel shape.
  :return: Location information in the format (image index, scale, y-offset,
     x-offset), where scale and y- and x-offsets identify the S2 unit from which
     the prototype was "imprinted".
  :rtype: 4 element array of int

  """
  return exp.extractor.prototype_algorithm.locations[kwidth][prototype]

def GetPrototype(exp, prototype, kwidth=0):
  """Return an S2 prototype from the experiment.

  :param int prototype: Index of S2 prototype.
  :param int kwidth: Index of kernel shape.

  """
  model = exp.extractor.model
  if model is None:
    raise ExpError("Experiment has no model")
  if model.s2_kernels is None:
    raise ExpError("Experiment has no S2 prototypes")
  return model.s2_kernels[kwidth][prototype]

def GetImagePatchForImprintedPrototype(exp, prototype=0, kwidth=0):
  """Get the image patch used to create a given imprinted prototype.

  :param int prototype: Index of S2 prototype.
  :param int kwidth: Index of kernel shape.
  :rtype: 2d-array of float
  :returns: image data used to construct given imprinted prototype

  """
  loc = exp.extractor.prototype_algorithm.locations[kwidth, prototype]
  image = exp.corpus.paths[loc[0]]
  model = exp.extractor.model
  y0,y1,x0,x1 = GetS2ReceptiveField(exp.extractor.model.params, *loc[1:],
      kwidth=kwidth)
  st = BuildLayer(model, Layer.IMAGE, model.MakeState(image))
  img = st[Layer.IMAGE]
  return img[y0:y1,x0:x1]

def GetBestPrototypeMatch(exp, image, prototype=0, kwidth=0):
  """Find the S2 unit with maximum response for a given prototype and image.

  :param image: Path to image on disk, or index of image in experiment.
  :param int prototype: Index of S2 prototype.
  :param int kwidth: Index of kernel shape.
  :rtype: 3-tuple of int
  :returns: S2 unit, given as (scale, y, x)

  """
  if not isinstance(image, basestring):
    image = exp.corpus.paths[image]
  model = exp.extractor.model
  st = BuildLayer(model, Layer.S2, model.MakeState(image), save_all=False)
  data = st[Layer.S2]  # indexed by (scale, kwidth, proto, height, width)
  scale = np.argmax([d[kwidth][prototype].max() for d in data])
  scale_data = data[scale][kwidth][prototype]
  y,x = [ c[0] for c in np.where(scale_data == scale_data.max()) ]
  return scale, y, x

def _ScaleImage(exp, image, scale=0):
  image = toimage(image, mode = 'F')
  scale_factor = 1. / exp.extractor.model.params.scale_factor
  size = np.array(image.size, np.int) * scale_factor**scale
  image = image.resize(np.round(size).astype(int), Image.ANTIALIAS)
  return fromimage(image)

def GetImageActivity(exp, image, scale=0):
  """Returns image layer for a given image and scale.

  :param image: Path to image on disk, or index of image in experiment.
  :param int scale: Index of scale band.
  :rtype: 2d array of float
  :return: Image data.

  """
  if not isinstance(image, basestring):
    image = exp.corpus.paths[image]
  model = exp.extractor.model
  st = BuildLayer(model, Layer.S1, model.MakeState(image), save_all=True)
  return _ScaleImage(exp, st[Layer.IMAGE], scale)

def _GetEvaluation(exp, evaluation):
  if evaluation >= len(exp.evaluation):
    raise ExpError("Missing evaluation information (record %d)" % evaluation)
  return exp.evaluation[evaluation]

def GetTrainingSet(exp, evaluation=0):
  """Returns the experiment's training set.

  The extractor determines the training set by checking the experiment's
  ``corpus.training_set`` attribute, and then randomly constructing a set if
  needed. The constructed set is stored in the experiment's
  ``extractor.training_set`` attribute.

  The evaluator determines the training set by checking the experiment's
  ``extractor.training_set`` attribute, then its ``corpus.training_set``, and
  then randomly constructing a set if needed. The constructed set is stored in
  the ``training_set`` attribute of the experiment's new evaluation record
  created by the evaluator.

  :param int evaluation: Index of the evaluation record to use.
  :rtype: 1D array of bool
  :return: Mask array indicating which images are in the training set.

  """
  if exp.corpus.training_set is not None:
    return exp.corpus.training_set
  if exp.extractor.training_set is not None:
    return exp.extractor.training_set
  if len(exp.evaluation) > 0:
    return _GetEvaluation(evaluation).training_set
  return None

def GetPredictions(exp, training=False, evaluation=0):
  """Get information about classifier predictions.

  :param exp: Experiment data.
  :param bool training: Return information about training images.
     Otherwise, information about the test set is returned.
  :param int evaluation: Index of the evaluation record to use.
  :rtype: list of 3-tuple of str
  :return: filename, true label, and predicted label for each image in the set

  """
  if len(exp.evaluation) < 1:
    raise ExpError("An evaluation record is required")
  training_set = GetTrainingSet(exp, evaluation=evaluation)
  r = exp.evaluation[evaluation].results
  predictions = r.training_predictions if training else r.predictions
  cnames = exp.corpus.class_names
  return zip(exp.corpus.paths[training_set],
      cnames[exp.corpus.labels[training_set]], cnames[predictions])

def GetEvaluationLayers(exp, evaluation=0):
  """Returns the model layers from which features were extracted.

  :param int evaluation: Index of the evaluation record to use.
  :rtype: list of str
  :return: Names of layers used for evaluation.

  """
  return [ l.name for l in _GetEvaluation(exp, evaluation).layers ]

def GetEvaluationResults(exp, evaluation=0):
  """Returns the results of a model evaluation.

  :param int evaluation: Index of the evaluation record to use.
  :rtype: :class:`glimpse.util.data.Data`
  :return: Result data, with attributes that depend on the method of evaluation.
     In general, the `feature_builder`, `score`, `score_func` attributes will be
     available.

  """
  return _GetEvaluation(exp, evaluation).results

def ShowS2Activity(exp, image, scale=0, prototype=0, kwidth=0):
  """Plot the S2 activity for a given image.

  :param image: Path to image on disk, or index of image in experiment.
  :param int scale: Index of scale band to use.
  :param int prototype: Index of S2 prototype to use.
  :param int kwidth: Index of kernel shape.

  """
  Show2dArray(exp.extractor.activation[image][Layer.S2][scale][kwidth][
      prototype])

def AnnotateS2Activity(exp, image, scale=0, prototype=0, kwidth=0):
  """Plot the S2 activity and image data for a given image.

  This shows the image in the background, with the S2 activity on top.

  :param image: Path to image on disk, or index of image in experiment.
  :param int scale: Index of scale band to use.
  :param int prototype: Index of S2 prototype to use.
  :param int kwidth: Index of kernel shape.

  """
  import matplotlib.pyplot as plt
  if not isinstance(image, basestring):
    image = exp.corpus.paths[image]
  model = exp.extractor.model
  # TODO: copy the model and use only the one given prototype.
  st = BuildLayer(model, Layer.S2, model.MakeState(image), save_all=True)
  image = _ScaleImage(exp, st[Layer.IMAGE], scale)
  s2 = st[Layer.S2][scale][kwidth][prototype]
  params = model.params
  left = bottom = (params.s2_kwidth[0]/2 * params.c1_sampling +
      params.c1_kwidth/2) * params.s1_sampling + params.s1_kwidth/2
  scale_factor = params.s2_sampling * params.c1_sampling * params.s1_sampling
  top = bottom + s2.shape[0] * scale_factor
  right = left + s2.shape[1] * scale_factor
  plt.imshow(image, cmap=plt.cm.gray)
  plt.imshow(s2, alpha=.5, extent=(left,right,top,bottom), cmap=plt.cm.RdBu_r)
  plt.xticks(())
  plt.yticks(())

def _AnnotateS2ReceptiveField(exp, image, scale, s2_y, s2_x, kwidth=0):
  import matplotlib.pyplot as plt
  if not isinstance(image, basestring):
    image = exp.corpus.paths[image]
  model = exp.extractor.model
  y0,y1,x0,x1 = GetS2ReceptiveField(exp.extractor.model.params, scale, s2_y,
      s2_x, kwidth_offset=kwidth)
  st = BuildLayer(model, Layer.IMAGE, model.MakeState(image))
  img = toimage(st[Layer.IMAGE])
  plt.imshow(img, cmap=plt.cm.gray, origin='lower')
  plt.xticks(())
  plt.yticks(())
  # invert y-coordinates, since origin='lower' in imshow
  y0 = img.size[1] - y0
  y1 = img.size[1] - y1
  plt.gca().add_patch(plt.Rectangle((x0,y0), width=x1-x0, height=y1-y0,
      fill=True, color='r', alpha=.5))

def AnnotateBestPrototypeMatch(exp, image, prototype=0, kwidth=0):
  """Plot the location that best matches a given S2 prototype.

  This shows the image in the background, with a red box over the region
  elliciting maximal response.

  :param image: Path to image on disk, or index of image in experiment.
  :param int prototype: Index of S2 prototype to use.
  :param int kwidth: Index of kernel shape.

  """
  if not isinstance(image, basestring):
    image = exp.corpus.paths[image]
  args = GetBestPrototypeMatch(exp, image, prototype, kwidth)
  _AnnotateS2ReceptiveField(exp, image, *args, kwidth=kwidth)

def AnnotateImprintedPrototype(exp, prototype=0, kwidth=0):
  """Plot the image region used to construct a given imprinted prototype.

  This shows the image in the background, with a red box over the imprinted
  region.

  :param int prototype: Index of S2 prototype to use.
  :param int kwidth: Index of kernel shape.

  """
  if not (hasattr(exp.extractor, 'prototype_algorithm') and
      hasattr(exp.extractor.prototype_algorithm, 'locations')):
    raise ExpError("Experiment does not use imprinted prototypes")
  loc = exp.extractor.prototype_algorithm.locations[kwidth, prototype]
  image = exp.corpus.paths[loc[0]]
  _AnnotateS2ReceptiveField(exp, image, *loc[1:], kwidth=kwidth)

def ShowPrototype(exp, prototype, kwidth=0):
  """Plot the prototype activation.

  There is one plot for each orientation band.

  :param int prototype: Index of S2 prototype to use.
  :param int kwidth: Index of kernel shape.

  """
  Show3dArray(GetPrototype(exp, prototype, kwidth))

def ShowC1Activity(exp, image, scale=0):
  """Plot the C1 activation for a given image.

  There is one plot for each orientation band.

  :param image: Path to image on disk, or index of image in experiment.
  :param int scale: Index of scale band to use.

  """
  Show3dArray(exp.extractor.activation[image][Layer.C1][scale])

def AnnotateC1Activity(exp, image, scale=0):
  """Plot the C1 activation for a given image.

  This shows the image in the background, with the activation plotted on top.
  There is one plot for each orientation band.

  :param image: Path to image on disk, or index of image in experiment.
  :param int scale: Index of scale band to use.

  """
  import matplotlib.pyplot as plt
  if not isinstance(image, basestring):
    image = exp.corpus.paths[image]
  model = exp.extractor.model
  st = BuildLayer(model, Layer.C1, model.MakeState(image), save_all=True)
  image = _ScaleImage(exp, st[Layer.IMAGE], scale)
  c1 = st[Layer.C1][scale]
  params = model.params
  left = bottom = (params.c1_kwidth/2) * params.s1_sampling + params.s1_kwidth/2
  scale_factor = params.c1_sampling * params.s1_sampling
  top = bottom + c1.shape[-2] * scale_factor
  right = left + c1.shape[-1] * scale_factor
  for i in range(len(c1)):
    plt.subplot(2, int(math.ceil(len(c1)/2)), i+1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.imshow(c1[i], alpha=.5, extent=(left,right,top,bottom),
        cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())

def ShowS1Activity(exp, image, scale=0):
  """Plot the S1 activation for a given image.

  There is one plot for each orientation band.

  :param image: Path to image on disk, or index of image in experiment.
  :param int scale: Index of scale band to use.

  """
  Show3dArray(exp.extractor.activation[image][Layer.S1][scale,orientation])

def AnnotateS1Activity(exp, image, scale=0):
  """Plot the S1 activation for a given image.

  This shows the image in the background, with the activation plotted on top.
  There is one plot for each orientation band.

  :param image: Path to image on disk, or index of image in experiment.
  :param int scale: Index of scale band to use.

  """
  import matplotlib.pyplot as plt
  if not isinstance(image, basestring):
    image = exp.corpus.paths[image]
  model = exp.extractor.model
  st = BuildLayer(model, Layer.S1, model.MakeState(image), save_all=True)
  image = _ScaleImage(exp, st[Layer.IMAGE], scale)
  s1 = st[Layer.S1][scale]
  params = model.params
  left = bottom = params.s1_kwidth/2
  scale_factor = params.s1_sampling
  top = bottom + s1.shape[-2] * scale_factor
  right = left + s1.shape[-1] * scale_factor
  for i in range(len(s1)):
    plt.subplot(2, int(math.ceil(len(s1)/2)), i+1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.imshow(s1[i], alpha=.5, extent=(left,right,top,bottom),
        cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())

def ShowS1Kernels(exp):
  """Plot the S1 Gabor kernels.

  There is one plot for each orientation band.

  """
  model = exp.extractor.model
  if model is None:
    raise ExpError("Experiment has no model")
  # grab filters for first phase across all orientations
  kernels = model.s1_kernels[:,0]
  Show3dArray(kernels)

def GetCorpusByName(name):
  """Return the path to a sample corpus of images.

  :param str name: Corpus name. One of 'easy', 'moderate', or 'hard'.
  :rtype: str
  :return: Filesystem path to corpus root directory.

  """
  import os
  path = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(path, '..', 'corpora', 'data', name)
  path = os.path.abspath(path)
  if not os.path.isdir(path):
    raise ExpError("Corpus not found: %s" % name)
  return path
