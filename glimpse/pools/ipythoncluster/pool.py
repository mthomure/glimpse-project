# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

from IPython.parallel import Client
import logging

#: Cached callable object.
_cfunc = None

def _mapper((func, id_, value)):
  return id_, func(value)

def _actual_map(view, func, iterable, progress, chunksize):
  if progress is None:
    map_results = view.map(func, iterable, block=False, ordered=False,
        chunksize=chunksize)
    return map_results.get(), map_results.metadata
  iterable = [ (func, id_, value) for id_, value in enumerate(iterable) ]
  p = progress(maxval=len(iterable)).start()
  p.update(0)
  map_results = view.map(_mapper, iterable, block=False, ordered=False,
      chunksize=chunksize)
  # reassemble results corresponding to input order
  results = [None] * len(iterable)
  for progress_idx, (id_, result) in enumerate(map_results):
    p.update(progress_idx)
    results[id_] = result
  p.finish()
  return results, map_results.metadata

class IPythonPool(object):
  """A worker pool that uses multiple cores from a set of remote machines."""

  # NOTE: naming for chunksize (rather than 'chunk_size') chosen to match
  # argument to LoadBalancedView.map().
  def __init__(self, profile=None, cached=True, save_timing=False,
      chunksize=None, maxchunksize=None):
    """

    :param bool save_timing: Store timing information in the `timing` member
       variable, and log the engine balance to level INFO.

    """
    self._client = c = Client(profile=profile)
    self._dview = c.direct_view()
    self._lbview = c.load_balanced_view()
    self.cached = cached
    self.save_timing = save_timing
    self.timing = None
    self.chunksize = chunksize
    self.maxchunksize = maxchunksize

  def map(self, func, iterable, progress=None, chunksize=None):
    """Apply a function to a list using multiple cores on remote machines."""
    iterable = list(iterable)  # make sure we know the length
    num_engines = len(self._client.ids)
    num_elements = len(iterable)
    if chunksize is None:
      if self.chunksize is None:
        chunksize = max(1, num_elements / (4*num_engines))
      else:
        chunksize = self.chunksize
    chunksize = min(chunksize, self.maxchunksize or chunksize)
    logging.info("IPythonPool using a chunk size of %d" % chunksize)
    if self.cached:
      # Copy callable object to all nodes, and evaluate.
      self._dview.apply_sync(_cfunc_setter, func)
      result, metadata = _actual_map(self._lbview, _cfunc_caller, iterable,
                           progress=progress, chunksize=chunksize)
      self._dview.apply_sync(_cfunc_setter, None)
    else:
      # Include original callable object in task messages.
      result, metadata = _actual_map(self._lbview, func, iterable,
          progress=progress, chunksize=chunksize)
    if self.save_timing:
      records = [ dict((k,md[k]) for k in ('engine_id', 'completed',
          'completed', 'msg_id', 'received', 'started', 'submitted'))
          for md in metadata ]
      # Get host/process info for engines
      einfo = self._dview.apply_async(_get_worker_info).get_dict()
      for tmg in records:
        tmg['host'],tmg['pid'] = einfo.get(tmg['engine_id'], (None,None))
      self.timing = einfo,records
      logging.info("Cluster balance: %f", engine_balance(self.timing))
    return result

def _get_worker_info():
  import os
  return os.uname()[1], os.getpid()

def _cfunc_setter(f):
  global _cfunc
  _cfunc = f

def _cfunc_caller(*args, **kw):
  global _cfunc
  return _cfunc(*args, **kw)

def MakePool(**kw):
  """Create a new IPython Cluster Pool.

  See `IPythonPool.__init__` for arguments. If available, the default profile,
  default chunksize, and maximum chunksize are read from the
  `GLIMPSE_IPYTHON_PROFILE`, `GLIMPSE_IPYTHON_CHUNKSIZE`, and
  `GLIMPSE_IPYTHON_MAXCHUNKSIZE` environment variables, respectively.

  """
  import os
  if 'profile' not in kw:
    kw['profile'] = os.environ.get('GLIMPSE_IPYTHON_PROFILE')
  if 'chunksize' not in kw:
    chunksize = os.environ.get('GLIMPSE_IPYTHON_CHUNKSIZE')
    if chunksize is not None:
      kw['chunksize'] = int(chunksize)
  if 'maxchunksize' not in kw:
    maxchunksize = os.environ.get('GLIMPSE_IPYTHON_MAXCHUNKSIZE')
    if maxchunksize is not None:
      kw['maxchunksize'] = int(maxchunksize)
  return IPythonPool(**kw)

def entropy(p, norm=False):
  import numpy as np
  p = np.array(p)
  assert (p >= 0).all()
  if norm:
    p = p / float(p.sum())
  else:
    assert abs(1 - p.sum()) < .001
  p = p[p > 0]  # p(x)=0 implies p(x)*log(p(x))=0
  return - np.dot(p, np.log(p)).sum()

def engine_balance(timing):
  """Calculate the workload balance across engines.

  An imbalance in workload across engines is reflected by a decrease in
  entropy of the number of jobs per engine. This assumes engines have a
  similar clock speed.

  """
  import numpy as np
  engine_info,records = timing
  engines = np.array([ r['engine_id'] for r in records ])
  all_engines = engine_info.keys()
  balance = entropy([(engines == engine).sum()
      for engine in all_engines], norm=True)
  # Report balance as normalized value (relative to perfectly balanced
  # set of N engines).
  balance /= np.log(len(all_engines))
  return balance

def PrintTiming(timing):
  import numpy as np
  engine_info,records = timing
  all_hosts = np.unique([v[0] for v in engine_info.values()])
  all_hosts.sort()
  all_engines = engine_info.keys()
  hosts = np.array([ r['host'] for r in records ])
  engines = np.array([ r['engine_id'] for r in records ])
  used_hosts = np.unique(hosts)
  used_engines = np.unique(engines)
  all_engs_per_host = np.array([ len([k for k,v in engine_info.items() if v[0] == host]) for host in all_hosts ])
  balance = engine_balance(timing)
  if len(records) < len(all_engines):
    balance_msg = "(more engines than images)"
  else:
    balance_msg = ""
  used_engs_per_host = np.array([len(np.unique(engines[hosts == host])) for host in all_hosts])  # number of engines per host
  msgs_per_host = np.array([(hosts == host).sum() for host in all_hosts])  # number of messages per host
  unused_engs_per_host = all_engs_per_host - used_engs_per_host

  print "Processed %d messages using %d engines" % (len(records), len(np.unique(engines)))
  print "Workload imbalance: %d%% %s" % ((1 - balance) * 100, balance_msg)

  # TODO: refactor following code into:
  # PrintTable(all_hosts, norm_msgs_per_host, norm_used_engs_per_host, norm_unused_engs_per_host, engs_per_host,
  #            titles=("HOST", "IMGS", "IMGS%", "ACTIVE%", "INACTIVE%", "ENGINES"))

  print "%-10s  %s  %s  %s" % ("HOST", "IMGS%", "ENGS%", "IDLE%")
  print "  ".join("-"*x for x in (10, 5, 5, 5))
  norm_msgs_per_host = msgs_per_host / float(msgs_per_host.sum()) * 100
  norm_used_engs_per_host = used_engs_per_host / float(len(all_engines)) * 100
  norm_unused_engs_per_host = unused_engs_per_host / float(len(all_engines)) * 100
  print "\n".join("%-10s  %5d  %5d  %5d" % ss for ss in zip(all_hosts,
                                                                #msgs_per_host,
                                                                norm_msgs_per_host,
                                                                #used_engs_per_host,
                                                                norm_used_engs_per_host,
                                                                #unused_engs_per_host,
                                                                norm_unused_engs_per_host))
  print "  ".join("-"*x for x in (10, 5, 5, 5))
  print "%-10s   100%%   %3d%%   %3d%%" % ("TOTAL",
                                       used_engs_per_host.sum() / float(len(all_engines)) * 100,
                                       unused_engs_per_host.sum() / float(len(all_engines)) * 100)

def Main():
  # Print information about ipython engine usage.
  import sys
  import cPickle as pickle
  if len(sys.argv) < 2:
    sys.exit("usage: %s EXP.dat" % sys.argv[0])
  with open(sys.argv[1], 'r') as fh:
    exp = pickle.load(fh)
  timing = exp.extractor.timing
  del exp
  PrintTiming(timing)

if __name__ == '__main__':
  Main()
