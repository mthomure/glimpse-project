# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from .config import ConfigException
from glimpse.util import zmq_cluster
from .pool import PoolWorker
from glimpse.pools import MulticorePool
from glimpse.util.zmq_cluster import Connect
import logging
from multiprocessing import Process
import os
import socket
import struct
import sys
import time
import zmq

def _LaunchTarget(launcher, config, name):
  context = zmq.Context()
  logging.info("Launching device: %s" % (name,))
  launcher(context, getattr(config, "%s_frontend" % name),
      getattr(config, "%s_backend" % name))

def LaunchBrokers(config):
  """Start a set of intermediate devices for the cluster on this machine.
  RETURN This method does not return."""

  def Launch(launcher, *names):
    for name in names:
      sender = getattr(config, "%s_sender" % name)
      receiver = getattr(config, "%s_receiver" % name)
      frontend = getattr(config, "%s_frontend" % name)
      backend = getattr(config, "%s_backend" % name)
      if sender.url == receiver.url:
        raise ConfigException("Must set different URLs for send and receive "
            "sockets to use intermediate device for %s messages." % name)
      if sender.bind or receiver.bind:
        raise ConfigException("Must not bind send or receive sockets when "
            "using ntermediate device for %s messages." % name)
      target_args = (launcher, config, name) #frontend, backend)
      Process(target = _LaunchTarget, args = target_args).start()

  if config.HasCommandChannels():
    Launch(zmq_cluster.LaunchForwarderDevice, "command", "command_response")
  Launch(zmq_cluster.LaunchStreamerDevice, "request", "result")

class EventLogger(object):

  RESPONSE_START = "COMMAND_RESPONSE_START"
  RESPONSE_STOP = "COMMAND_RESPONSE_STOP"
  RESPONSE_PING = "COMMAND_RESPONSE_PING"

  def __init__(self, context, sender):
    self.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    self.publisher = sender.MakeSocket(context, type = zmq.PUB)
    self.elapsed_time = 0  # total time spent processing requests
    self.num_requests = 0  # number of requests processed

  def Prefix(self):
    return socket.getfqdn(), os.getpid(), self.start_time

  def LogStart(self):
    bitwidth = 8 * struct.calcsize("P")
    version = sys.version_info
    if isinstance(version, tuple):  # old style of version_info
      version = version[:3]
    else:  # new style
      version = version.major, version.minor, version.micro
    payload = self.Prefix(), os.uname(), version, bitwidth
    self.publisher.send_pyobj((self.RESPONSE_START, payload))

  def LogStop(self, exit_status):
    payload = self.Prefix(), exit_status
    self.publisher.send_pyobj((self.RESPONSE_STOP, payload))

  def ReplyPing(self):
    payload = self.Prefix(), self.elapsed_time, self.num_requests
    self.publisher.send_pyobj((self.RESPONSE_PING, payload))

class Worker(PoolWorker):

  # kill the worker, but sys.exit with -1. this can be used to request a "nanny"
  # script to relaunch the worker.
  CMD_KILL_ERROR = "CLUSTER_WORKER_KILL_ERROR"
  # reply with basic worker information.
  CMD_PING = "CLUSTER_WORKER_PING"

  def __init__(self, context, config, receiver_timeout = None, pool = None):
    super(Worker, self).__init__(context, config, receiver_timeout, pool)
    self.event_logger = EventLogger(context, config.command_response_sender)
    self.exit_status = 0

  def Setup(self):
    super(Worker, self).Setup()

  def HandleRequest(self, request):
    start_time = time.time()
    result = super(Worker, self).HandleRequest(request)
    self.event_logger.elapsed_time += time.time() - start_time
    self.event_logger.num_requests += 1
    return result

  def HandleCommand(self, command):
    if command == Worker.CMD_KILL_ERROR:
      self.exit_status = -1
      finish = True
    elif command == Worker.CMD_PING:
      self.event_logger.ReplyPing()
      finish = False
    else:
      finish = super(Worker, self).HandleCommand(command)
    logging.info("Worker: got command %r, finish=%s" % (command, finish))
    return finish

  def Run(self):
    self.event_logger.LogStart()
    super(Worker, self).Run()
    self.event_logger.LogStop(self.exit_status)

  @staticmethod
  def PingWorkers(command_sender, command_response_receiver, wait_time = None):
    """Determine the set of active workers.
    command_sender -- (zmq.socket or Connect)
    command_response_receiver -- (zmq.socket or Connect)
    wait_time -- (int) how long to wait for replies (in seconds)
    RETURN (iterator) node information for responding workers
    """
    if wait_time == None:
      wait_time = 1  # wait for one second by default
    logging.info("PoolWorker: sending ping command")
    logging.info("PoolWorker:   command: %s" % command_sender)
    logging.info("PoolWorker:   responses: %s" % command_response_receiver)
    context = zmq.Context()
    if isinstance(command_sender, Connect):
      commands = command_sender.MakeSocket(context, type = zmq.PUB)
    else:
      commands = command_sender
    if isinstance(command_response_receiver, Connect):
      responses = command_response_receiver.MakeSocket(context, type = zmq.SUB,
          options = {zmq.SUBSCRIBE : ""})
    else:
      responses = command_response_receiver
    poller = zmq.Poller()
    poller.register(responses, zmq.POLLIN)
    time.sleep(1)  # Wait for workers to connect. This is necessary to make sure
                   # all subscribers get the QUIT message.
    commands.send_pyobj(Worker.CMD_PING)
    poll_timeout = 1000  # poll for one second
    wait_start_time = time.time()
    while time.time() - wait_start_time < wait_time:
      # Wait for input, with timeout given in milliseconds
      if poller.poll(timeout = poll_timeout):
        type_, payload = responses.recv_pyobj()
        if type_ == EventLogger.RESPONSE_PING:
          prefix, elapsed_time, num_requests = payload
          name, pid, start_time = prefix
          if num_requests > 0:
            elapsed_time /= num_requests
          elapsed_time = "%.2f" % elapsed_time
          yield name, pid, start_time, elapsed_time, num_requests
    raise StopIteration

def LaunchWorker(config, num_processes = None):
  """Start a cluster worker on the local host.
  config -- (ClusterConfig) socket configuration for the cluster
  num_processes -- (int) number of sub-processes to use
  RETURN This method calls sys.exit(), so does not return.
  """
  if num_processes != None:
    num_processes = int(num_processes)
  pool = MulticorePool(num_processes)
  worker = Worker(zmq.Context(), config,
      receiver_timeout = None,  # wait indefinitely for task requests
      pool = pool)
  worker.Setup()  # connect/bind sockets and prepare for work
  worker.Run()  # run the request/reply loop until termination
  sys.exit(worker.exit_status)

def PingWorkers(config, wait_time = None):
  """Determine the set of active workers."""
  if wait_time:
    wait_time = int(wait_time)
  if not config.HasCommandChannels():
    raise ConfigException("No URL found for sending command messages. Update "
        "your cluster configuration.")
  for node in Worker.PingWorkers(config.command_sender,
      config.command_response_receiver, wait_time):
    print " ".join(map(str, node))

def KillWorkers(config):
  """Send a QUIT command to any workers running on the cluster."""
  if not config.HasCommandChannels():
    raise ConfigException("No URL found for sending command messages. Update "
        "your cluster configuration.")
  Worker.SendKillCommand(zmq.Context(), config.command_sender)
  time.sleep(1)  # wait for ZMQ to flush message queues

def RestartWorkers(config):
  """Send KILL-ERROR to any workers running on cluster, causing a restart."""
  if not config.HasCommandChannels():
    raise ConfigException("No URL found for sending command messages. Update "
        "your cluster configuration.")
  Worker.SendKillCommand(zmq.Context(), config.command_sender,
      Worker.CMD_KILL_ERROR)
  time.sleep(1)  # wait for ZMQ to flush message queues

def FlushSocket(socket):
  while True:
    try:
      socket.recv(zmq.NOBLOCK)
    except zmq.ZMQError, e:
      if e.errno == zmq.EAGAIN:
        break
      else:
        raise

def FlushCluster(config):
  """Consume any queued messages waiting on the cluster."""
  ctx = zmq.Context()
  request_socket = config.request_receiver.MakeSocket(ctx, type = zmq.PULL)
  result_socket = config.result_receiver.MakeSocket(ctx, type = zmq.PULL)
  time.sleep(1)  # wait for connections
  FlushSocket(request_socket)
  FlushSocket(result_socket)
