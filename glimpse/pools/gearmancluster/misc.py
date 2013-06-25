import cPickle
import gearman

class PickleDataEncoder(gearman.DataEncoder):
  """A data encoder that reads and writes using the Pickle format.

  By default, a :class:`GearmanClient` can only send byte-strings. If we want to
  be able to send Python objects, we must specify a data encoder. This will
  automatically convert byte strings to Python objects (and vice versa) for
  *all* commands that have the 'data' field. See
  http://gearman.org/index.php?id=protocol for client commands that send/receive
  'opaque data'.

  """

  @classmethod
  def encode(cls, encodable_object):
    return cPickle.dumps(encodable_object, protocol = 2)

  @classmethod
  def decode(cls, decodable_string):
    return cPickle.loads(decodable_string)

#: Gearman task ID for the *map* task.
GEARMAN_TASK_MAP = "glimpse_map"
#: Instructs the worker to exit with signal 0.
COMMAND_QUIT = "quit"
#: Instructs the worker to exit with signal -1.
COMMAND_RESTART = "restart"
#: Instructs the worker to respond with processing statistics.
COMMAND_PING = "ping"
#: Instructs the worker to add data to its stateful data store.
COMMAND_SETMEMORY = "set-memory"
#: Instructs the worker to clear its stateful data store.
COMMAND_CLEARMEMORY = "clear-memory"
#: Variable name in worker memory for cached function.
CACHED_FUNC_KEY = "_cached_func"
