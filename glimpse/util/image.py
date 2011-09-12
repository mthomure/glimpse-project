
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Functions for dealing with images.
#

import numpy
import sys

def ImageToArray(img, array = None, transpose = True):
  """Load image data into a 2D numpy array. If array is unspecified, then one
  will be generated automatically. Note that this array may not be contiguous.
  The array holding image data is returned."""
  def MakeBuffer():
    if img.mode == 'L':
      return numpy.empty(img.size, dtype = numpy.uint8)
    elif img.mode == 'RGB':
      return numpy.empty(img.size + (3,), dtype = numpy.uint8)
    elif img.mode == 'F':
      return numpy.empty(img.size, dtype = numpy.float)
    elif img.mode == '1':
      return numpy.empty(img.size, dtype = numpy.bool)
    raise Exception("Can't load data from image with mode: %s" % img.mode)
  def CopyImage(dest):
    img_data = img.load()
    for idx in numpy.ndindex(img.size):
      dest[idx] = img_data[idx]
    return dest
  def CheckArrayShape():
    shape = list(img.size)
    if transpose:
      shape = shape[::-1]
    shape = tuple(shape)
    assert shape == array.shape, "Array has wrong shape: expected %s but got %s" % (shape, array.shape)
  if array != None:
    if not transpose:
      CheckArrayShape()
      return CopyImage(array)
    else:
      CheckArrayShape()
      # copy image to new buffer, copy buffer.T to array
      array[:] = CopyImage(MakeBuffer()).T
      return array
  else:
    if not transpose:
      return CopyImage(MakeBuffer())
    else:
      return CopyImage(MakeBuffer()).T
  assert False, "Internal logic error!"

def ShowImage(img, fname = None):
  if sys.platform == "darwin":
    img.show()
  else:
    ShowImageOnLinux(img, fname)

def ShowImageOnLinux(img, fname = None):
  dir = TempDir()
  if not fname or '..' in fname:
    fname = 'img.png'
  path = dir.MakePath(fname)
  img.save(path)
  RunCommand("eog -n %s" % path, False, False)

