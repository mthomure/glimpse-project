# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import itertools
import logging
import os
import socket
import time
import zmq

def SocketTypeToString(type):
  mapping = { zmq.PUSH : 'PUSH', zmq.PULL : 'PULL',
      zmq.PUB : 'PUB', zmq.SUB : 'SUB', zmq.REQ : 'REQ', zmq.REP : 'REP' }
  return mapping[type]

class ReceiverTimeoutException(Exception):
  """Indicates that a ZMQ recv() command timed out."""
  pass

class WorkerException(Exception):
  """Indicates that a worker node reported an exception while processing a
  request."""

  worker_exception = None  # the exception object thrown in the worker process

class FutureSocket(object):
  """Describes the options needed to connect/bind a ZMQ socket to an
  end-point."""

  url = None  # (required) the URL passed to the connect() or bind() method of
              # the created socket
  type = None  # (optional) the type of socket to create
  bind = False  # (optional) whether this socket connects or binds
  options = None  # (optional) a dictionary of socket options (see setsockopt())
  pre_delay = None  # (optional) amount of time (in seconds) to wait before
                    # connecting/binding the socket
  post_delay = None  # (optional) amount of time (in seconds) to wait after
                     # connecting/binding the socket

  def __init__(self, url = None, type = None, bind = False, options = None):
    self.url, self.type, self.bind, self.options = url, type, bind, options

  def __str__(self):
    d = dict(self.__dict__.items())
    if d['type'] != None:
      d['type'] = SocketTypeToString(d['type'])
    keys = filter((lambda k: d[k] != None), d.keys())
    values = map(self.__getattribute__, keys)
    return "Connect(%s)" % ", ".join("%s=%s" % x for x in zip(keys, values))

  def __repr__(self):
    return str(self)

  def __call__(self, context, url = None, type = None, bind = None,
      options = None):
    return self.MakeSocket(context, url, type, bind, options)

  def MakeSocket(self, context, url = None, type = None, bind = None,
      options = None, pre_delay = None, post_delay = None):
    """Create the socket. Arguments take precendence over their corresponding
    object attributes."""
    if type == None:
      type = self.type
    if url == None:
      url = self.url
    if bind == None:
      bind = self.bind
    if options == None:
      options = {}
    if self.options != None:
      if options == None:
        options = self.options
      else:
        options = dict(options.items() + self.options.items())
    if pre_delay == None:
      pre_delay = self.pre_delay
    if post_delay == None:
      post_delay = self.post_delay
    assert type is not None
    socket = context.socket(type)
    for k, v in options.items():
      socket.setsockopt(k, v)
    if pre_delay != None:
      time.sleep(pre_delay)
    assert url is not None
    if bind:
      logging.info("Binding %s socket to %s with context %s" % (
          SocketTypeToString(type), url, hash(context)))
      socket.bind(url)
    else:
      logging.info("Connecting %s socket to %s with context %s" % (
          SocketTypeToString(type), url, hash(context)))
      socket.connect(url)
    if post_delay != None:
      time.sleep(post_delay)
    return socket

Connect = FutureSocket

def InitSocket(context, connect_or_socket, type = None, **kwargs):
  if isinstance(connect_or_socket, FutureSocket):
    connect_or_socket = connect_or_socket(context, type = type, **kwargs)
  else:
    if type != None and connect_or_socket.socket_type != type:
      raise ValueError("Expected socket of type %s, but got type %s" % (
          SocketTypeToString(type),
          SocketTypeToString(connect_or_socket.socket_type)))
  return connect_or_socket

def MakeSocket(context, url, type, bind = False, options = None):
  return FutureSocket(url = url, type = type, bind = bind, options = options) \
      .MakeSocket(context)

class BasicVentilator(object):
  """Push tasks to worker nodes on the cluster."""

  request_sender = None  # (required) constructor for the writer socket

  def __init__(self, context, request_sender, worker_connect_delay = None):
    """
    worker_connect_delay -- (float) length of time to wait (in seconds) between
                            calling Setup() and starting to transmit tasks to
                            the cluster. This gives worker nodes a chance to
                            connect, avoiding ZMQ's "late joiner syndrome".
                            Default delay is one second.
    """
    if worker_connect_delay == None:
      worker_connect_delay = 1.
    self.request_sender = request_sender
    self.ready = False
    self.start = None
    self.context = context
    self.worker_connect_delay = worker_connect_delay
    self.num_total_requests = 0

  def Setup(self):
    logging.info("BasicVentilator: starting ventilator on pid %d" % os.getpid())
    logging.info("BasicVentilator:   sender: %s" % self.request_sender)
    if isinstance(self.request_sender, Connect):
      self.sender = self.request_sender.MakeSocket(self.context,
          type = zmq.PUSH)
    else:
      self.sender = self.request_sender
    self._connect_delay = time.time() + self.worker_connect_delay
    logging.info("BasicVentilator: bound, starting at pid %d" % os.getpid())
    self.ready = True

  def Shutdown(self):
    del self.sender
    self.ready = False

  def Send(self, requests):
    """Reads tasks from an iterator, and sends them to worker nodes.
    RETURN the number of sent tasks.
    """
    # Set up the connection
    if not self.ready:
      self.Setup()
    # Give worker nodes time to connect
    time_delta = time.time() - self._connect_delay
    if time_delta > 0:
      time.sleep(time_delta)
    logging.info("BasicVentilator: starting send")
    num_requests = 0
    for request in requests:
#      logging.info("BasicVentilator: sending task %d at time %s" % \
#          (self.num_total_requests, time.time()))
      self.sender.send_pyobj(request)
      self.num_total_requests += 1
      num_requests += 1
    logging.info("BasicVentilator: finished sending %d tasks" % num_requests)
    return num_requests

class BasicSink(object):
  """Collect results from worker nodes on the cluster."""

  result_receiver = None  # (required) constructor for the reader socket
  command_receiver = None  # (optional) constructor for the command socket

  CMD_KILL = "CLUSTER_SINK_KILL"  # Send this to the command socket to shut down
                                  # the sink.

  def __init__(self, context, result_receiver, command_receiver = None,
      receiver_timeout = None):
    """Create a new Sink object.
    result_receiver -- (zmq.socket or Connect) channel on which to receive
                       results
    command_receiver -- (zmq.socket or Connect, optional) channel on which to
                        receive quit command
    receiver_timeout -- (int) time to wait for a result before quiting
    """
    self.context = context
    self.result_receiver = result_receiver
    self.command_receiver = command_receiver
    self.receiver_timeout = receiver_timeout
    self._ready = False

  def Setup(self):
    self._poller = zmq.Poller()
    if isinstance(self.result_receiver, Connect):
      self._receiver_socket = self.result_receiver.MakeSocket(self.context,
          type = zmq.PULL)
    else:
      self._receiver_socket = self.result_receiver
    self._poller.register(self._receiver_socket)
    self._command_socket = None
    logging.info("BasicSink: starting sink on pid %d" % os.getpid())
    logging.info("BasicSink:   reader: %s" % self.result_receiver)
    logging.info("BasicSink:   command: %s" % self.command_receiver)
    if self.command_receiver != None:
      if isinstance(self.command_receiver, Connect):
        self._command_socket = self.command_receiver.MakeSocket(self.context,
            type = zmq.SUB)
      else:
        self._command_socket = self.command_receiver
      self._poller.register(self._command_socket)
    logging.info("BasicSink: bound at pid %d" % os.getpid())
    self._ready = True
    logging.info("BasicSink: setup done")

  def Shutdown(self):
    del self._poller
    del self._receiver_socket
    del self._command_socket

  def Receive(self, num_results = None, timeout = None):
    """Returns an iterator over results, available as they arrive at the sink.
    Raises a ReceiverTimeoutException if no result arrives in the given time
    period.
    num_results -- (int) return after a fixed number of results have arrived
    timeout -- (int) time to wait for each result (in milliseconds)
    """
    if not self._ready:
      self.Setup()
    idx = 0
    if timeout == None:
      timeout = self.receiver_timeout
    while True:
      if num_results != None and idx >= num_results:
        break
      socks = dict(self._poller.poll(timeout))
      if len(socks) == 0:
        raise ReceiverTimeoutException
      if self._receiver_socket in socks:
        result = self._receiver_socket.recv_pyobj()
        yield result
        idx += 1
      if self._command_socket in socks:
        cmd = self._command_socket.recv_pyobj()
        logging.info("BasicSink: got command %s on pid %d" % (cmd, os.getpid()))
        if cmd == self.CMD_KILL:
          logging.info("BasicSink: received quit command")
          break
        # Ignore unrecognized commands.
    raise StopIteration

  @staticmethod
  def SendKillCommand(context, command_sender):
    """
    command_sender -- (zmq.socket or Connect)
    """
    #~ logging.info("Sink:   command: %s" % command_sender.url)
    if isinstance(command_sender, Connect):
      commands = command_sender.MakeSocket(context, type = zmq.PUB)
    else:
      commands = command_sender
    time.sleep(1)  # wait for sink process/thread to connect
    logging.info("BasicSink.SendKillCommand: sending kill command")
    commands.send_pyobj(BasicSink.CMD_KILL)
    logging.info("BasicSink.SendKillCommand: kill command sent")

class ClusterRequest(object):
  """A cluster request, corresponding to the input value of a callback."""

  payload = None  # task's input values.
  metadata = None  # optional information associated with the request. this
                   # information is copied to the result object.

  def __init__(self, payload = None, metadata = None):
    self.payload = payload
    self.metadata = metadata

class ClusterResult(object):
  """A cluster result, corresponding to the output value of a callback when
  applied to one input element."""

  status = None  # whether the input elements were processed successfully
  payload = None  # output corresponding to task's input elements. this will
                  # either be a list -- in the case of a map() operation -- or a
                  # scalar -- in the case of a reduce().

  request_metadata = None  # optional information that was associated with
                           # request.
  metadata = None  # optional information associated with result.
  exception = None  # exception that occurrred during processing, if any

  STATUS_SUCCESS = "OK"  # indicates that request was processed successfully
  STATUS_FAIL = "FAIL"  # indicates that error occurred while processing request

  def __init__(self, status = None, payload = None, request_metadata = None,
      metadata = None, exception = None):
    self.status, self.payload, self.request_metadata, self.metadata, \
        self.exception = status, payload, request_metadata, metadata, exception

class Ventilator(BasicVentilator):

  def Send(self, requests, metadata = None):
    """Send requests to worker nodes.
    requests -- (iterable) callback arguments
    metadata -- (iterable) same number of metadata objects
    """
    # Wrap in a cluster request with an empty ID.
    if metadata != None:
      requests = itertools.imap(ClusterRequest, requests, metadata)
    else:
      requests = itertools.imap(ClusterRequest, requests)
    return super(Ventilator, self).Send(requests)

class Sink(BasicSink):

  def Receive(self, num_results = None, timeout = None, metadata = False):
    results = super(Sink, self).Receive(num_results, timeout)
    for result in results:
      if result.status != ClusterResult.STATUS_SUCCESS:
        raise WorkerException("Caught exception in worker node (%s:%s)\n%s" % \
            (result.metadata[0], result.metadata[1], result.exception))
      if metadata:
        yield result.payload, result.request_metadata, result.metadata
      else:
        yield result.payload
    raise StopIteration

class BasicWorker(object):

  CMD_KILL = "CLUSTER_WORKER_KILL"  # Send this to the command socket to shut
                                    # down the worker.

  def __init__(self, context, request_receiver, result_sender,
      command_receiver = None, receiver_timeout = None):
    """Handles requests that arrive on a socket, writing results to another
    socket.
    context -- (zmq.Context) context used to create sockets
    request_receiver -- (Connect) channel for receiving incoming requests
    result_sender -- (Connect) channel for sending results
    command_receiver -- (zmq.socket or Connect) channel for receiving commands
    receiver_timeout -- (int) how long to wait for a request or command before
                        quiting. If not set, wait indefinitely.
    """
    self.context, self.request_receiver, self.result_sender, \
        self.command_receiver, self.receiver_timeout = context, \
        request_receiver, result_sender, command_receiver, \
        receiver_timeout
    self.receiver = None

  def Setup(self):
    logging.info("BasicWorker: starting worker on pid %s at %s" % (os.getpid(),
        time.asctime()))
    logging.info("BasicWorker:   receiver: %s" % self.request_receiver)
    logging.info("BasicWorker:   command: %s" % self.command_receiver)
    logging.info("BasicWorker:   sender: %s" % self.result_sender)
    # Set up the sockets
    self.receiver = self.request_receiver.MakeSocket(self.context,
        type = zmq.PULL)
    self.sender = self.result_sender.MakeSocket(self.context, type = zmq.PUSH)
    self.poller = zmq.Poller()
    self.poller.register(self.receiver, zmq.POLLIN)
    if self.command_receiver != None:
      self.cmd_subscriber = self.command_receiver.MakeSocket(self.context,
          type = zmq.SUB, options = {zmq.SUBSCRIBE : ""})
      self.poller.register(self.cmd_subscriber, zmq.POLLIN)
    else:
      self.cmd_subscriber = None
    logging.info("BasicWorker: bound at pid %d" % os.getpid())

  def Run(self):
    if self.receiver == None:
      self.Setup()
    # Handle incoming requests, and watch for KILL commands
    while True:
      socks = dict(self.poller.poll(self.receiver_timeout))
      if len(socks) == 0:
        raise ReceiverTimeoutException
      if self.receiver in socks:
        request = self.receiver.recv_pyobj()
        # Full metadata for a successful result includes the full hostname and
        # PID of the worker process, and the time elapsed while computing the
        # result.
        result_metadata = socket.getfqdn(), os.getpid()
        result = ClusterResult(request_metadata = request.metadata,
            metadata = result_metadata)
        try:
          # Apply user request_handler to the request
          start_time = time.time()
          result.payload = self.HandleRequest(request.payload)
          # Add the time consumed by handling the request
          result.metadata = result.metadata + (time.time() - start_time,)
          result.status = ClusterResult.STATUS_SUCCESS
        except Exception, e:
          logging.info(("BasicWorker: caught exception %s from " % e) + \
              "request processor")
          result.exception = e
          result.status = ClusterResult.STATUS_FAIL
        self.sender.send_pyobj(result)
      if self.cmd_subscriber in socks:
        cmd = self.cmd_subscriber.recv_pyobj()
        logging.info("BasicWorker: got cmd %s on pid %d" % (cmd, os.getpid()))
        if self.HandleCommand(cmd):
          logging.info("BasicWorker: quiting on pid %d" % os.getpid())
          break

  def HandleRequest(self, request):
    return request

  def HandleCommand(self, command):
    finish = False
    if command == BasicWorker.CMD_KILL:
      logging.info("BasicWorker: received kill command")
      finish = True
    return finish

  @staticmethod
  def SendKillCommand(context, command_sender, command = None):
    """Send a kill command to all workers on a given channel.
    command_sender -- (zmq.socket or Connect)
    command -- message to send to workers. defaults to CMD_KILL.
    """
    if command == None:
      command = BasicWorker.CMD_KILL
    logging.info("BasicWorker: sending kill command")
    logging.info("BasicWorker:   command: %s" % command_sender)
    if isinstance(command_sender, Connect):
      command_sender = command_sender.MakeSocket(context, type = zmq.PUB)
    time.sleep(1)  # Wait for workers to connect. This is necessary to make sure
                   # all subscribers get the QUIT message.
    command_sender.send_pyobj(command)
    logging.info("BasicWorker: sent kill command")

def LaunchStreamerDevice(context, frontend_connect, backend_connect):
  frontend = frontend_connect.MakeSocket(context, type = zmq.PULL, bind = True)
  backend = backend_connect.MakeSocket(context, type = zmq.PUSH, bind = True)
  logging.info("LaunchStreamerDevice: starting streamer on pid %d" % \
      os.getpid())
  logging.info("LaunchStreamerDevice:   frontend: %s" % (frontend_connect,))
  logging.info("LaunchStreamerDevice:   backend: %s" % (backend_connect,))
  zmq.device(zmq.STREAMER, frontend, backend)

def LaunchForwarderDevice(context, frontend_connect, backend_connect):
  frontend = frontend_connect.MakeSocket(context, type = zmq.SUB, bind = True,
      options = {zmq.SUBSCRIBE : ""})
  backend = backend_connect.MakeSocket(context, type = zmq.PUB, bind = True)
  logging.info("LaunchForwarderDevice: starting forwarder on pid %d" % \
      os.getpid())
  logging.info("LaunchForwarderDevice:   frontend: %s" % (frontend_connect,))
  logging.info("LaunchForwarderDevice:   backend: %s" % (backend_connect,))
  zmq.device(zmq.FORWARDER, frontend, backend)
