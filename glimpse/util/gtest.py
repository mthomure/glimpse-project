# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from contextlib import contextmanager
import importlib
import numpy as np
import os
from scipy.misc import toimage
import shutil
import tempfile
import types
import unittest

__all__ = [
    'unittest',
    'TempDir',
    'TempFile',
    'MonkeyPatch',
    'RecordedFunctionCall',
    'MakeCorpus',
    'MakeCorpusWithImages',
    ]

@contextmanager
def TempDir():
  """Context manager that creates a temporary directory on disk.

  The created directory is deleted from disk when the interpreter leaves the
  context.

  Example:
  >>> with TempDir() as dname:
  >>>   print dname

  """
  path = tempfile.mkdtemp()
  yield path
  shutil.rmtree(path)

@contextmanager
def TempFile(suffix=''):
  """Context manager that creates a temporary file on disk.

  The created file is deleted from disk when the interpreter leaves the
  context.

  Example:
  >>> with TempFile() as fname:
  >>>   print fname

  """
  fd, path = tempfile.mkstemp(suffix=suffix)
  yield path
  os.remove(path)

@contextmanager
def MonkeyPatch(*records):
  """Context manager that temporarily replace function definitions.

  :param records: Functions to replace, and the callable with which to replace
     them.
  :type records: 3-tuple or list of 3-tuple

  A record is a 3-tuple of (MODULE, SYMBOL, FUNC), where
  :param MODULE: a module type or a string giving the module name
  :param str SYMBOL: symbol name to replace
  :param callable FUNC: function to call when SYMBOL is accessed

  All patched symbols are reset when the interpreter leaves the context.

  Example:
  >>> import glimpse.experiment
  >>> with MonkeyPatch(glimpse.experiment, 'Verbose', (lambda x=None: None)):
  >>>   glimpse.experiment.Verbose()
  >>> with MonkeyPatch(('glimpse.experiment', 'Verbose',
          (lambda x=None: None)),):
  >>>   glimpse.Experiment.Verbse()

  """
  assert len(records) > 0
  if (isinstance(records[0], basestring) or
      isinstance(records[0], types.ModuleType) or
      not hasattr(records[0], '__len__')):
    records = [ records ]  # assume single record was given
  unwind_records = list()
  for module_name, symbol, patch_value in records:
    if isinstance(module_name, types.ModuleType):
      mod = module_name
    else:
      mod = importlib.import_module(module_name)
    orig_value = getattr(mod, symbol)
    setattr(mod, symbol, patch_value)
    unwind_records.append((mod, symbol, orig_value))
  yield
  for mod, symbol, orig_value in unwind_records:
    setattr(mod, symbol, orig_value)

class RecordedFunctionCall(object):
  """A constant-valued function that records information about it's call."""

  def __init__(self, return_value=None):
    self.called = False  # whether function has been called
    self.args = None  # positional arguments passed to function
    self.kw = None  # keyword arguments passed to function
    self.return_value = return_value  # value that function will return

  def __call__(self, *args, **kw):
    self.called = True
    self.args = args
    self.kw = kw
    return self.return_value

def MakeCorpus(root, **subdirs):
  """Create a corpus on disk with empty files.

  :param subdirs: file names in sub-directory
  :type subdirs: list of str

  Example:
  >>> MakeCorpus('/tmp/my_test_root', cats=('cat1.png', 'cat2.png'),
          dogs=('dog1.png', 'dog2.png'))

  """
  for cls, files in subdirs.items():
    os.mkdir(os.path.join(root, cls))
    for f in files:
      with open(os.path.join(root, cls, f), 'w') as fh:
        fh.write("1")

def MakeCorpusWithImages(root, **subdirs):
  """Create a corpus on disk with random image data.

  :param subdirs: file names in sub-directory
  :type subdirs: list of str

  Example:
  >>> MakeCorpusWithImages('/tmp/my_test_root', cats=('cat1.png', 'cat2.png'),
          dogs=('dog1.png', 'dog2.png'))

  """
  for cls, files in subdirs.items():
    os.mkdir(os.path.join(root, cls))
    for f in files:
      img = toimage(np.random.random((100,101)) * 255)
      img.save(os.path.join(root, cls, f))
