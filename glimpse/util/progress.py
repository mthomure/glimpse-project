# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

try:
  import progressbar as _pb
except ImportError:
  _pb = None

class MockProgressBar:
  """A progress updater that ignores all inputs."""
  def __init__(self, maxval=None):
    pass
  def start(self):
    return self
  def update(self, value=None):
    pass
  def finish(self):
    pass
  def __call__(self, iterable):
    return iterable

def ProgressBar(*args, **kw):
  if _pb is None:
    return MockProgressBar()
  return _pb.ProgressBar(*args, widgets=[_pb.ETA(), '   ', _pb.Bar(),
      '   Speed:', _pb.FileTransferSpeed(unit='unit')], **kw)
