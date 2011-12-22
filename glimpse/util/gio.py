
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Functions related to ASCII and pickled-binary file I/O.
#

import cPickle
from cStringIO import StringIO
import numpy
import pprint
import struct
import sys
from misc import IsIterable, IsString

def ReadLines(fname_or_fh):
  """Returns lines of file as array."""
  if isinstance(fname_or_fh, str):
    fh = open(fname_or_fh)
    lines = [ s.strip() for s in fh ]
    fh.close()
  else:
    fh = fname_or_fh
    lines = [ s.strip() for s in fh ]
  return lines

def WriteLines(lines, fname):
  fh = open(fname, 'w')
  for l in lines:
    print >>fh, l
  fh.close()

def WriteString(string, fname):
  fh = open(fname, 'w')
  print >>fh, string
  fh.close()

def ReadMatrix(fname, sep=" "):
  return [ s.split(sep) for s in ReadLines(fname) ]

def WriteMatrix(matrix, fname, sep=" "):
  WriteLines(sep.join(matrix), fname)

def ReadGTiffFile(fname):
  """Read a file in GTiff format as a 3D numpy array."""
  ## Assumes elements have type float32
  from osgeo import gdal
  from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
  ds = gdal.Open(fname, GA_ReadOnly)
  if ds == None:
    raise Exception("Error while reading GTiff file(%s)" % fname)
  num_bands = ds.RasterCount
  height = ds.RasterYSize
  width = ds.RasterXSize
  data = numpy.empty((num_bands, height, width), dtype=numpy.float32)
  for band_idx in range(num_bands):
    band = ds.GetRasterBand(band_idx + 1)
    for y in range(band.YSize):
      scanline = band.ReadRaster(0,
                                 y,
                                 band.XSize,
                                 1,
                                 band.XSize,
                                 1,
                                 GDT_Float32)
      data[band_idx,y,:] = struct.unpack('f' * band.XSize, scanline)
  return data

### Serialization ###

ENCODING_PICKLE = "p"
ENCODING_CSV = "c"   # CSV-formatted numeric data
ENCODING_FREE_TEXT = "s"
INPUT_ENCODINGS = (ENCODING_PICKLE, ENCODING_CSV)
OUTPUT_ENCODINGS = (ENCODING_PICKLE, ENCODING_CSV, ENCODING_FREE_TEXT)
CSV_DELIM = None  # Use any whitespace string

def LoadAll(fnames, encoding = ENCODING_PICKLE):
  """Return an iterator over all objects in a set of files."""
  if not IsIterable(fnames) or IsString(fnames):
    fnames = (fnames,)
  for fh in fnames:
    do_close = False
    if not hasattr(fh, 'read'):
      fh = open(fh, 'r')
      do_close = True
    if encoding == ENCODING_PICKLE:
      source = cPickle.Unpickler(fh)
      while True:
        try:
          yield source.load()
        except EOFError:
          break
        except cPickle.UnpicklingError:
          raise Exception("Caught exception (cPickle.UnpicklingError): maybe input encoding should not be \"%s\"?" % ENCODING_PICKLE)
    elif encoding == ENCODING_CSV:
      try:
        data = []
        for line in ReadLines(fh):
          record = []
          for field in line.strip().split(CSV_DELIM):
            try:
              record.append(float(field))
            except ValueError:
              record.append(field)
          data.append(record)
        yield numpy.array(data)
      except ValueError, e:
        raise Exception("Caught exception (ValueError): maybe input encoding should not be \"%s\"?" % ENCODING_CSV)
    else:
      raise Exception("Invalid encoding: %s" % encoding)
    if do_close:
      fh.close()

def LoadPython(fname):
  """Load a file of Python source as a dictionary.
  fname -- name of file from which to load options
  """
  d = {}
  # Evaluate python source file, using d as namespace.
  execfile(fname, {}, d)
  return d

def LoadByFileName(fname, *args):
  """Load data either from a python source file (if fname ends in ".py"), or a
  file containing a single pickled variable.
  fname -- name of file from which to load options
  """
  if IsString(fname) and fname.endswith('.py'):
    return LoadPython(fname)
  return Load(fname, *args)

def Load(fname = sys.stdin, encoding = ENCODING_PICKLE):
  """Return the first object in a file."""
  return LoadAll((fname,), encoding).next()

def Store(obj, fname = sys.stdout, encoding = ENCODING_PICKLE):
  if hasattr(fname, 'write'):
    fh = fname
  else:
    fh = open(fname, 'w')
  if encoding == ENCODING_PICKLE:
    cPickle.dump(obj, fh, 2)
  elif encoding == ENCODING_CSV:
    delim = CSV_DELIM or " "
    assert(IsIterable(obj)), "Failed to write data -- not an array/matrix"
    for record in obj:
      if IsIterable(record):
        print >>fh, delim.join(map(str, record))
      else:
        print >>fh, record
  elif encoding == ENCODING_FREE_TEXT:
    pprint.pprint(obj, fh)
  else:
    raise Exception("Unknown encoding: %s" % encoding)
  if fh != fname:
    fh.close()

def SuppressStdout(callback, *args, **vargs):
  """Evaluate a function, ignoring the data written to stdout. This works even
  if the code emitting data to stdout is in a compiled extension.
  """
  # Copied from http://stackoverflow.com/questions/4178614/suppressing-output-
  # of-module-calling-outside-library
  so, sys.stdout = sys.stdout, StringIO()
  ret = callback(*args, **vargs)
  sys.stdout = so
  return ret
