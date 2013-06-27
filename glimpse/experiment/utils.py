"""Low-level, stateless functions used by glab experiment functions."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from collections import Counter
from contextlib import contextmanager
from itertools import chain
import logging
import numpy as np
import os
import time

from glimpse.pools import MakePool
from glimpse.util.garray import FlattenArrays
from glimpse.util import learn

@contextmanager
def TimedLogging(level, msg):
  start = time.time()
  yield
  duration = time.time() - start
  logging.log(level, "%s took %.3fs" % (msg, duration))

class DirReader(object):
  """Read directory contents."""

  def __init__(self, ignore_hidden = True):
    #: Ignore hidden directory entries (i.e., those starting with '.').
    self.ignore_hidden = ignore_hidden

  @staticmethod
  def _HiddenPathFilter(path):
    # Ignore "hidden" entries in directory.
    return not path.startswith('.')

  def _Read(self, dir_path):
    entries = os.listdir(dir_path)
    if self.ignore_hidden:
      entries = filter(DirReader._HiddenPathFilter, entries)
    return [ os.path.join(dir_path, entry) for entry in entries ]

  def ReadDirs(self, dir_path):
    """Read list of sub-directories.

    :param str dir_path: Filesystem for directory to read.
    :rtype: list of str
    :return: List of sub-directories in directory.

    """
    return filter(os.path.isdir, self._Read(dir_path))

  def ReadFiles(self, dir_path):
    """Read list of files.

    :param str dir_path: Filesystem for directory to read.
    :rtype: list of str
    :return: List of files in directory.

    """
    return filter(os.path.isfile, self._Read(dir_path))

def ReadCorpusDirs(dirs, reader):
  """Create list of paths and class labels for a set of image directories.

  :param dirs: Directory for each image class.
  :type dirs: list of str
  :param reader: Filesystem reader.

  """
  if len(dirs) == 0:
    raise Exception("Must specify one or more sub-directories")
  paths = map(sorted, map(reader.ReadFiles, dirs))
  counts = map(len, paths)
  for p, c in zip(paths, counts):
    if c == 0:
      raise Exception("No images found in directory -- %s" % p)
  labels = ([i]*c for i, c in enumerate(counts))  # class labels are 0-based
  labels = np.array(list(chain(*labels)))
  paths = np.array(list(chain(*paths)))  # flatten lists
  if len(paths) == 0:
    raise Exception("No images found")
  return paths, labels

def BalanceCorpus(labels, shuffle=False):
  """Choose a subset of instances, such that the resulting corpus is balanced.

  A balanced corpus has the same number of instances for each class.

  :type labels: 1D array of int
  :param labels: Instance class labels.
  :param bool shuffle: Whether to randomly choose the included instances. By
     default, instances in the tail of the list are always dropped.
  :rtype: 1D array of bool
  :returns: Mask of chosen instances.

  """
  label_counts = Counter(labels)
  min_count = min(label_counts.values())
  mask = np.zeros(len(labels), dtype = np.bool)
  for label, count in label_counts.items():
    idx = np.where(labels == label)[0]
    if shuffle:
      np.random.shuffle(idx)
    mask[idx[:min_count]] = True
  return mask

def ExtractFeatures(layers, states):
  """Create feature vectors from the activation maps for a set of images.

  :param layers: Model layers to use when constructing feature vector.
  :type layers: LayerSpec or tuple of LayerSpec
  :param features: Model activity for a set of images.
  :type features: list of BaseState

  :rtype: 2D ndarray of float
  :returns: 1D feature vector for each image

  """
  if not hasattr(layers, '__len__') or isinstance(layers, basestring):
    layers = (layers,)
  if len(layers) < 1:
    raise ValueError("Must specify one or more layers from which to build "
        "features")
  # Flatten entire dataset to 1D vector
  result = FlattenArrays([[ state[layer] for layer in layers ]
      for state in states ])
  # Partition feature set by image
  result = result.reshape(len(states), -1)
  return result
