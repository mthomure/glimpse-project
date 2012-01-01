# Thin wrapper around progressbar library
#   http://code.google.com/p/python-progressbar/
# Provides dummy wrapper if library is unavailable.

try:
  from progressbar import *
except ImportError:
  class ProgressBar(object):
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def update(self, x): pass
    def finish(self): pass
