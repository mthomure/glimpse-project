from glimpse import util
from IPython.parallel import Client
from IPython.parallel.error import CompositeError, NoEnginesRegistered

def KillWorkers(client):
  """Kill all active workers."""
  client.shutdown(restart = True, block = True, hub = False)

def RestartWorkers(client):
  """Kill and relaunch all active workers."""
  # The ipython engine creates a file in the worker run directory to communicate the restart signal to the wrapper script.
  # This is a hackish stop-gap measure until client.shutdown responds to the 'restart' argument.
  dview = client.direct_view()
  def restart():
    import os, socket
    fname = os.path.join('run', 'glimpse-cluster', socket.getfqdn(), 'workers', 'restart')
    with file(fname, 'a'):
      os.utime(fname, None)
  dview.apply_sync(restart)
  client.shutdown(restart = False, block = True, hub = False)

def PingWorkers(client):
  """Determine the set of active workers."""
  def ping():
    import os, socket
    return socket.getfqdn(), os.getpid()
  dview = client.direct_view()
  try:
    results = dview.apply_sync(ping)
    print "\n".join(" ".join(map(str, r)) for r in results)
  except NoEnginesRegistered:
    print "--None--"

def main(args = None):
  methods = map(eval, ("KillWorkers", "RestartWorkers", "PingWorkers"))
  profile = None
  try:
    opts, args = util.GetOptions("p:v", args = args)
    for opt, arg in opts:
      if opt == '-p':
        profile = arg
      elif opt == '-v':
        import logging
        logging.getLogger().setLevel(logging.INFO)
    if len(args) < 1:
      raise util.UsageException
    method = eval(args[0])
    client = Client(profile = profile)
#    dview = client.direct_view()
    method(client, *args[1:])
  except util.UsageException, e:
    method_info = [ "  %s -- %s" % (m.func_name, m.__doc__.splitlines()[0])
        for m in methods ]
    util.Usage("[options] CMD [ARGS]\n"
        "  -p PROF   Set the IPython profile\n"
        "  -v        Be verbose with logging\n"
        "CMDs include:\n" + "\n".join(method_info),
        e)

if __name__ == "__main__":
  main()
