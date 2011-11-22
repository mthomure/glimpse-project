# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import ConfigParser
from glimpse.executors.executor import DynamicMap
from glimpse.executors.cluster_executor import ClusterExecutor
from glimpse import util
from glimpse.util import zmq_cluster
from glimpse.util.zmq_cluster import Connect
import itertools
import logging
from multiprocessing import Process
import os
import sys
import time

class ConfigException(Exception): pass

class ClusterConfig(object):
  """Reads the configuration for cluster sockets from an INI file. Recognized
  options include the following.

  [DEFAULT]

  # Default socket used to send and receive task requests.
  request_url = ...

  # Whether to bind (rather than connect) the request socket.
  request_bind = Boolean, default False

  # Default socket used to send and receive task results.
  request_url = ...

  # Whether to bind the result socket.
  result_bind = Boolean, default False

  # Default socket used to send and receive commands.
  command_url = ...

  # Whether to bind the command socket.
  command_bind = Boolean, default False

  [READER]

  # Socket used by worker to receive requests.
  request_url = ...

  # Whether to bind (rather than connect) the request socket. See the zmq
  # library for details.
  request_bind = Boolean, default False

  # Socket used by sink to receive results.
  request_url = ...

  # Whether to bind the result socket.
  result_bind = Boolean, default False

  # Socket used by sink and by workers to receive commands.
  command_url = ...

  # Whether to bind the sink/worker command socket.
  command_bind = Boolean, default False

  [WRITER]

  # Socket used by ventilator to send requests.
  request_url = String

  # Whether to bind (rather than connect) the ventilator's request socket. See
  # the zmq library for details.
  request_bind = Boolean, default False

  # Socket used by worker to send results.
  result_url = ...

  # Whether to bind the worker's result socket.
  result_bind = Boolean, default False

  # Outgoing socket used to send commands to sink and workers.
  command_url = ...

  # Whether to bind the outgoing command socket.
  command_bind = Boolean, default False

  [BROKER]

  # Socket used to receive requests from ventilator.
  request_frontend_url = String

  # Socket used to send requests to workers.
  request_backend_url = String

  # Socket used to receive results from workers.
  result_frontend_url = String

  # Socket used to send results to sink.
  result_backend_url = String

  # Socket used to receive commands from user.
  command_frontend_url = String

  # Socket used to send commands to workers and sink.
  command_backend_url = String
  """

  def __init__(self, *config_files):
    """Create a new object.
    config_files -- path to one or more cluster configuration files
    """
    self.config = ConfigParser.SafeConfigParser()
    # Setup the default, empty configuration
    self.config.set('DEFAULT', 'request_url', '')
    self.config.set('DEFAULT', 'result_url', '')
    self.config.set('DEFAULT', 'command_url', '')
    self.config.set('DEFAULT', 'request_bind', 'False')
    self.config.set('DEFAULT', 'result_bind', 'False')
    self.config.set('DEFAULT', 'command_bind', 'False')
    self.config.add_section('READER')
    self.config.add_section('WRITER')
    # Read the specified config files
    read_files = self.config.read(*config_files)
    if len(read_files) == 0:
      raise util.UsageException("Unable to read any socket configuration "
          "files.")
    # Validate config information
    for msg_type, direction in itertools.product(("request", "result"),
        ("sender", "receiver")):
      channel = getattr(self, "%s_%s" % (msg_type, direction))
      if channel.url == "":
        raise ConfigException("Configuration information is incomplete. "
            "Missing socket information for the %s %s." % (msg_type, direction))

  def _getConnect(self, section, name):
    return Connect(url = self.config.get(section, '%s_url' % name),
        bind = bool(self.config.getboolean(section, '%s_bind' % name)))

  def HasCommandChannel(self):
    return self.config.get('WRITER', 'command_url') != ""

  @property
  def request_sender(self):
    return self._getConnect('WRITER', 'request')

  @property
  def request_receiver(self):
    return self._getConnect('READER', 'request')

  @property
  def result_sender(self):
    return self._getConnect('WRITER', 'result')

  @property
  def result_receiver(self):
    return self._getConnect('READER', 'result')

  @property
  def command_sender(self):
    if self.config.get('WRITER', 'command_url') == "":
      return None
    return self._getConnect('WRITER', 'command')

  @property
  def command_receiver(self):
    if self.config.get('READER', 'command_url') == "":
      return None
    return self._getConnect('READER', 'command')

  def _getBrokerConnect(self, name):
    if not self.config.has_section('BROKER'):
      return None
    return Connect(url = self.config.get('BROKER', '%s_url' % name),
        bind = True)  # broker always binds

  @property
  def request_frontend(self):
    return self._getBrokerConnect('request_frontend')

  @property
  def request_backend(self):
    return self._getBrokerConnect('request_backend')

  @property
  def result_frontend(self):
    return self._getBrokerConnect('result_frontend')

  @property
  def result_backend(self):
    return self._getBrokerConnect('result_backend')

  @property
  def command_frontend(self):
    return self._getBrokerConnect('command_frontend')

  @property
  def command_backend(self):
    return self._getBrokerConnect('command_backend')

from glimpse.executors.multicore_executor import MulticoreExecutor
from glimpse.executors.executor import DynamicMap, DynamicMapRequestHandler
import zmq

def Worker(config, num_processes = None):
  """Start a cluster worker on this machine."""
  if num_processes != None:
    num_processes = int(num_processes)
  executor = MulticoreExecutor(DynamicMapRequestHandler, num_processes)

  def worker_callback(dynamic_batch_request):
    """Convert a dynamic batch request to a batch result."""
    function, batch_request = dynamic_batch_request
    # Set up the iterator, which (when evaluated) will apply the multicore
    # executor to the elements of the batch request.
    logging.info("glimpse_cluster.worker_callback: dynamic mapping function "
        "across %d elements" % len(batch_request))
    result_it = DynamicMap(executor, function, batch_request)
    result_list = list(result_it)  # evaluate the iterator
    return result_list  # return batch result

  receiver_timeout = None  # if set, makes worker quit when no available tasks
  worker = zmq_cluster.Worker(zmq.Context(), config.request_receiver,
      config.result_sender, worker_callback,
      command_receiver = config.command_receiver,
      receiver_timeout = receiver_timeout)
  worker.Setup()  # connect/bind sockets and prepare for work
  worker.Run()  # run the request/reply loop until termination

def KillWorkers(config):
  """Kill any workers running on the cluster."""
  command_sender = config.command_sender
  if command_sender == None:
    raise ConfigException("No URL found for sending command messages. Update "
        "your cluster configuration.")
  zmq_cluster.Worker.SendKillCommand(zmq.Context(), command_sender)
  time.sleep(1)  # wait for ZMQ to flush message queues

def LaunchRequestStreamer(config):
  context = zmq.Context()
  logging.info("Launching device: request streamer")
  zmq_cluster.LaunchStreamerDevice(context, config.request_frontend,
      config.request_backend)

def LaunchResultStreamer(config):
  context = zmq.Context()
  logging.info("Launching device: result streamer")
  zmq_cluster.LaunchStreamerDevice(context, config.result_frontend,
      config.result_backend)

def LaunchCommandForwarder(config):
  context = zmq.Context()
  logging.info("Launching device: command forwarder")
  zmq_cluster.LaunchForwarderDevice(context, config.command_frontend,
      config.command_backend)

def Brokers(config):
  """Start a set of intermediate devices for the cluster on this machine."""
  if config.HasCommandChannel():
    if config.command_sender.url == config.command_receiver.url:
      raise ConfigException("Must set different URLs for send and receive "
          "sockets to use intermediate device for command messages.")
    if config.command_sender.bind or config.command_receiver.bind:
      raise ConfigException("Must not bind send or receive sockets when using "
          "intermediate device for command messages.")
    Process(target = LaunchCommandForwarder, args = (config,)).start()
  if config.request_sender.url == config.request_receiver.url:
    raise ConfigException("Must set different URLs for send and receive "
        "sockets to use intermediate device for request messages.")
  if config.request_sender.bind or config.request_receiver.bind:
    raise ConfigException("Must not bind send or receive sockets when using "
        "intermediate device for request messages.")
  Process(target = LaunchRequestStreamer, args = (config,)).start()
  if config.result_sender.url == config.result_receiver.url:
    raise ConfigException("Must set different URLs for send and receive "
        "sockets to use intermediate device for result messages.")
  if config.result_sender.bind or config.result_receiver.bind:
    raise ConfigException("Must not bind send or receive sockets when using "
        "intermediate device for result messages.")
  LaunchResultStreamer(config)

def TransformImages(config, *fnames):

  # use default model for now
  from glimpse.models.viz2.model import Model, Layer, State, ModelTransform
  from glimpse.models.misc import InputSource
  from glimpse.models.viz2.params import Params
  from glimpse.backends.cython_backend import CythonBackend
  model = Model(CythonBackend(), Params())
  layer = Layer.RETINA
  xform = ModelTransform(model, layer, save_all = False)
  input_states = map(State, map(InputSource, fnames))
  context = zmq.Context()
  executor = ClusterExecutor(context, config.request_sender,
      config.result_receiver, config.command_sender, config.command_receiver)
  executor.Setup()
  output_states = DynamicMap(executor, xform, input_states)

  output_states = list(output_states)

  #~ print input_states
  #~ print output_states

def Methods():
  return map(eval, ("Worker", "KillWorkers", "Brokers"))

def main():
  try:
    config_files = tuple()
    opts, args = util.GetOptions("c:v")
    for opt, arg in opts:
      if opt == '-c':
        config_files = config_files + (arg,)
      elif opt == '-v':
        import logging
        logging.getLogger().setLevel(logging.INFO)
    if len(args) < 1:
      raise util.UsageException
    if not config_files:
      raise util.UsageException("Must specify a socket configuration file.")
    method = eval(args[0])
    config = ClusterConfig(*config_files)
    method(config, *args[1:])
  except ConfigException, e:
    sys.exit("Configuration error: %s" % e)
  except util.UsageException, e:
    methods = [ "  %s -- %s" % (m.func_name, m.__doc__)
        for m in Methods() ]
    util.Usage("[options] CMD [ARGS]\n"
        "  -c FILE   Read socket configuration from FILE\n"
        "CMDs include:\n" + "\n".join(methods),
        e)

if __name__ == "__main__":
  main()
