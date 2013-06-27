# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from glimpse.util import pil_fix

import cPickle as pickle
import math
import numpy as numpy
np = numpy  # use numpy and np to allow flexibility in command string argument
from numpy import *
import os
import pprint
import re
import sys

from glimpse.util.option import *

from glimpse.experiment import *

def Load(fname):
  if hasattr(fname, 'read'):
    return pickle.load(fname)
  with open(fname, 'rb') as fh:
    return pickle.load(fh)

def Store(obj, fname):
  if hasattr(fname, 'write'):
    return pickle.dump(obj, fname, protocol = 2)
  with open(fname, 'wb') as fh:
    return pickle.dump(obj, fh, protocol = 2)

ENCODING_PICKLE, ENCODING_CSV, ENCODING_FREE_TEXT = 'p', 'c', 's'

def StoreAs(obj, fname, encoding):
  """Store a single object to a file."""
  if hasattr(fname, 'write'):
    fh = fname
  else:
    fh = open(fname, 'w')
  if encoding == ENCODING_PICKLE:
    Store(obj, fname)
  elif encoding == ENCODING_CSV:
    delim = " "
    assert hasattr(obj, '__iter__'), "Failed to write data -- not array/matrix"
    for record in obj:
      if hasattr(record, '__iter__'):
        print >>fh, delim.join(map(str, record))
      else:
        print >>fh, record
  elif encoding == ENCODING_FREE_TEXT:
    pprint.pprint(obj, fh)
  else:
    raise Exception("Unknown encoding: %s" % encoding)
  if fh != fname:
    fh.close()

def LoadAll(fnames, encoding = ENCODING_PICKLE, delim=' '):
  """Read all objects in a set of files.

  :param fnames: Set of files to read.
  :type fnames: list of str
  :param encoding: Encoding of input files.
  :returns: Iterator over the set of all objects read.

  """
  if (not hasattr(fnames, '__iter__') or isinstance(fnames, basestring) or
      isinstance(fnames, file)):
    fnames = (fnames,)
  for fh in fnames:
    do_close = False
    if not hasattr(fh, 'read'):
      fh = open(fh, 'r')
      do_close = True
    if encoding == ENCODING_PICKLE:
      source = pickle.Unpickler(fh)
      while True:
        try:
          yield source.load()
        except EOFError:
          break
        except pickle.UnpicklingError:
          raise Exception("Caught pickle error: maybe "
              "input encoding should not be \"%s\"?" % ENCODING_PICKLE)
    elif encoding == ENCODING_CSV:
      try:
        data = []
        for line in fh:
          record = []
          for field in line.strip().split(delim):
            try:
              record.append(float(field))
            except ValueError:
              record.append(field)
          data.append(record)
        yield np.array(data)
      except ValueError, e:
        raise Exception("Caught exception (ValueError): maybe input encoding "
            "should not be \"%s\"?" % ENCODING_CSV)
    else:
      raise Exception("Invalid encoding: %s" % encoding)
    if do_close:
      fh.close()

def Codex(opts, fnames):
  # Must bind all variable names not defined locally. Otherwise, we can't use
  # "bare" exec statement below.
  global Load, Store, core, math, numpy, np, os, re, sys
  load = Load
  store = Store
  fnames = map(lambda f: (f == "-" and sys.stdin or f), fnames)
  if len(fnames) < 1:
    fnames = [ sys.stdin ]
  # Do preprocessing activities
  if opts.packages is not None:
    for p in opts.packages:
      exec "import %s" % p
  # Evaluate the BEGIN command before processing the input array.
  if opts.begin_command:
    try:
      exec opts.begin_command
    except Exception, e:
      print >>sys.stderr, "Error evaluating BEGIN command: %s" % e
      sys.exit(-1)
  # Process each object in the input stream(s)
  _idx = 0
  for obj in LoadAll(fnames, opts.input_encoding):
    o = obj
    if opts.as_array:
      # Evaluate command on each element of input array.
      array = obj
      for obj in array:
        o = obj
        if opts.object_command:
          # Evaluate the command
          if opts.object_statement:
            exec opts.object_command
            result = o
          else:
            result = eval(opts.object_command)
        else:
          result = obj
        if not (result == None or opts.quiet):
          StoreAs(result, sys.stdout, opts.output_encoding)
        _idx += 1
    else:
      # Evaluate command once on the entire input array.
      if opts.object_command:
        if opts.object_statement:
          exec opts.object_command
          result = o
        else:
          result = eval(opts.object_command)
      else:
        result = obj
      if not (result == None or opts.quiet):
        StoreAs(result, sys.stdout, opts.output_encoding)
      _idx += 1
  # Evaluate the END command after processing the input array.
  if opts.end_command:
    try:
      exec opts.end_command
    except Exception, e:
      print >>sys.stderr, "Error evaluating END command: %s" % e
      sys.exit(-1)

def MakeCliOptions():
  return OptionRoot(
      Option('as_array', flag = 'a', doc = "Treat each input object as an "
          "array, over which whose elements are iterated"),
      Option('begin_command', flag = 'b:', doc = "Run command before iterating "
          "over object(s)"),
      Option('object_command', flag = 'c:', doc = "Apply command to unpickled "
          "objects, where 'o' is the current object and 'idx' is the current "
          "index."),
      Option('end_command', flag = 'e:', doc = "Run command after iterating "
          "over object(s)"),
      Option('input_encoding', flag = 'i:', default = ENCODING_PICKLE,
          doc = "Set encoding type for input stream(s) -- "
              "(%s) pickle, (%s) csv text, (%s) summary text" %
                  (ENCODING_PICKLE, ENCODING_CSV, ENCODING_FREE_TEXT)),
      Option('output_encoding', flag = 'o:', default = ENCODING_FREE_TEXT,
          doc = "Set encoding type for output stream"),
      Option('packages', flag = 'p:', doc = "Import a comma-delimited list of "
          "python packages"),
      Option('quiet', flag = 'q', doc = "Be quiet -- don't write result to "
          "output"),
      Option('object_statement', flag = 's', doc = "Treat object command as a "
          "statement, not an expression"),
      Option('help', flag = 'h', doc = "Print this help and exit"),
      )

def Main(argv = None):
  options = MakeCliOptions()
  try:
    fnames = ParseCommandLine(options, argv = argv)
    if options.help.value:
      print >>sys.stderr, "Usage: %s [options] [FILE ...]" % sys.argv[0]
      PrintUsage(options, stream = sys.stderr, width = 80)
      sys.exit(-1)
    Codex(OptValue(options), fnames)
  except OptionError, e:
    print >>sys.stderr, "Usage Error (use -h for help): %s." % e

if __name__ == '__main__':
  Main()
