from IPython.parallel import Client
from IPython.parallel.util import interactive

@interactive  # The direct view pushes to a particular namespace (i.e., not this module's NS). Thus, we must use this decorator for FUNC to be found in the correct namespace.
def func_int(*args):
  global FUNC, SHARED_DATA
  return FUNC(SHARED_DATA, *args)

class ClusterPool(object):

  def __init__(self, *args, **kwargs):
    self.client = Client(*args, **kwargs)
    self.lbview = self.client.load_balanced_view()
    self.chunksize = 1

  def map_with_shared_data(self, func, shared_data, args, chunksize = None):
    """Map a function over each in a set of arguments, also passing a constant shared variable to each invocation."""
    # no imap with shared data, since we couldn't guarantee the integrity of FUNC and SHARED_DATA
    self.dview.push(dict(FUNC = func, SHARED_DATA = shared_data), block = True)
    return self.lbview.map_sync(func_int, args, chunksize = chunksize or self.chunksize, ordered = True)

  def map(self, func, args, chunksize = None):
    map = self.lbview.map
    return iter(map(func, args, chunksize = chunksize or self.chunksize, block = True, ordered = True))

  def imap(self, func, args, chunksize = None):
    map = self.lbview.map
    return iter(map(func, args, chunksize = chunksize or self.chunksize, block = False, ordered = True))

  def imap_unordered(self, func, args, chunksize = None):
    map = self.lbview.map
    return iter(map(func, args, chunksize = chunksize or self.chunksize, block = False, ordered = False))

def MakePool(config_file = None, chunksize = None):
  if config_file != None or chunksize != None:
    logging.warn("ipython_cluster.MakePool: Ignoring config_file and/or chunksize arguments")
  return ClusterPool()
