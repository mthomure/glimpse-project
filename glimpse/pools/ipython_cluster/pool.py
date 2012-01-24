from IPython.parallel import Client

class ClusterPool(object):

  def __init__(self, *args, **kwargs):
    self.client = Client(*args, **kwargs)
    self.lbview = self.client.load_balanced_view()
    self.chunksize = 1

  def _map(self, func, args, chunksize = None, block = False):
    if chunksize == None:
      chunksize = self.chunksize or 1
    return self.lbview.map(func, args, block = block, chunksize = chunksize, ordered = ordered)

  def map(self, func, args, chunksize = None):
    map = self.lbview.map
    return iter(map(func, args, chunksize = chunksize or self.chunksize, block = True, ordered = True))

  def imap(self, func, args, chunksize = None):
    map = self.lbview.map
    return iter(map(func, args, chunksize = chunksize or self.chunksize, block = False, ordered = True))

  def imap_unordered(self, func, args, chunksize = None):
    map = self.lbview.map
    return iter(map(func, args, chunksize = chunksize or self.chunksize, block = False, ordered = False))
