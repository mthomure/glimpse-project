# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import ConfigParser
from glimpse.util.zmq_cluster import Connect
import itertools

class ConfigException(Exception):
  """This exception indicates that an error occurred while reading the cluster
  configuration."""
  pass

class ClusterConfig(object):
  """Reads the configuration for cluster sockets from an INI file. Recognized
  options include the following.

  [DEFAULT]

  # Default socket used to send and receive task requests.
  request_url = String

  # Whether to bind (rather than connect) the request socket.
  request_bind = Boolean, default False

  # Default socket used to send and receive task results.
  result_url = String

  # Whether to bind the result socket.
  result_bind = Boolean, default False

  # Default socket used to send and receive commands.
  command_url = String

  # Whether to bind the command socket.
  command_bind = Boolean, default False

  # Default socket used to send and receive command responses.
  command_response_url = String

  # Whether to bind the command-response socket.
  command_response_bind = Boolean, default False

  [READER]

  # Socket used by worker to receive requests.
  request_url = String

  # Whether to bind (rather than connect) the request socket. See the zmq
  # library for details.
  request_bind = Boolean, default False

  # Socket used by sink to receive results.
  request_url = String

  # Whether to bind the result socket.
  result_bind = Boolean, default False

  # Socket used by sink and by workers to receive commands.
  command_url = String

  # Whether to bind the incoming sink/worker command socket.
  command_bind = Boolean, default False

  # Socket used by sink and workers to send command responses.
  command_response_url = String

  # Whether to bind the outgoing sink/worker command-response socket.
  command_response_bind = Boolean, default False

  [WRITER]

  # Socket used by ventilator to send requests.
  request_url = String

  # Whether to bind (rather than connect) the ventilator's request socket. See
  # the zmq library for details.
  request_bind = Boolean, default False

  # Socket used by worker to send results.
  result_url = String

  # Whether to bind the worker's result socket.
  result_bind = Boolean, default False

  # Outgoing socket used to send commands to sink and workers.
  command_url = String

  # Whether to bind the outgoing command socket.
  command_bind = Boolean, default False

  # Incoming socket used to receive command responses from sink and workers.
  command_response_url = String

  # Whether to bind the incoming command socket.
  command_response_bind = Boolean, default False

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

  # Socket used to receive command responses from sink and workers.
  command_response_frontend_url = String

  # Socket used to send command responses to user.
  command_response_backend_url = String
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
    self.config.set('DEFAULT', 'command_response_url', '')
    self.config.set('DEFAULT', 'request_bind', 'False')
    self.config.set('DEFAULT', 'result_bind', 'False')
    self.config.set('DEFAULT', 'command_bind', 'False')
    self.config.set('DEFAULT', 'command_response_bind', 'False')
    self.config.add_section('READER')
    self.config.add_section('WRITER')
    # Read the specified config files
    read_files = self.config.read(config_files)
    if len(read_files) == 0:
      raise ConfigException("Unable to read any socket configuration "
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

  def HasCommandChannels(self):
    return self.config.get('WRITER', 'command_url') != "" and \
        self.config.get('WRITER', 'command_response_url') != ""

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

  @property
  def command_response_sender(self):
    if self.config.get('WRITER', 'command_response_url') == "":
      return None
    return self._getConnect('WRITER', 'command_response')

  @property
  def command_response_receiver(self):
    if self.config.get('READER', 'command_response_url') == "":
      return None
    return self._getConnect('READER', 'command_response')

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

  @property
  def command_response_frontend(self):
    return self._getBrokerConnect('command_response_frontend')

  @property
  def command_response_backend(self):
    return self._getBrokerConnect('command_response_backend')
