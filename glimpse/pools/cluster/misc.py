# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.pools.cluster.config import ConfigException
from glimpse.util import zmq_cluster
import logging
import multiprocessing
from multiprocessing import Process
import zmq

def KillWorkers(config):
  """Send a QUIT command to any workers running on the cluster."""
  command_sender = config.command_sender
  if command_sender == None:
    raise ConfigException("No URL found for sending command messages. Update "
        "your cluster configuration.")
  zmq_cluster.Worker.SendKillCommand(zmq.Context(), command_sender)
  time.sleep(1)  # wait for ZMQ to flush message queues

def _LaunchRequestStreamer(config):
  context = zmq.Context()
  logging.info("Launching device: request streamer")
  zmq_cluster.LaunchStreamerDevice(context, config.request_frontend,
      config.request_backend)

def _LaunchResultStreamer(config):
  context = zmq.Context()
  logging.info("Launching device: result streamer")
  zmq_cluster.LaunchStreamerDevice(context, config.result_frontend,
      config.result_backend)

def _LaunchCommandForwarder(config):
  context = zmq.Context()
  logging.info("Launching device: command forwarder")
  zmq_cluster.LaunchForwarderDevice(context, config.command_frontend,
      config.command_backend)

def LaunchBrokers(config):
  """Start a set of intermediate devices for the cluster on this machine.
  RETURN This method does not return."""
  if config.HasCommandChannel():
    if config.command_sender.url == config.command_receiver.url:
      raise ConfigException("Must set different URLs for send and receive "
          "sockets to use intermediate device for command messages.")
    if config.command_sender.bind or config.command_receiver.bind:
      raise ConfigException("Must not bind send or receive sockets when using "
          "intermediate device for command messages.")
    Process(target = _LaunchCommandForwarder, args = (config,)).start()
  if config.request_sender.url == config.request_receiver.url:
    raise ConfigException("Must set different URLs for send and receive "
        "sockets to use intermediate device for request messages.")
  if config.request_sender.bind or config.request_receiver.bind:
    raise ConfigException("Must not bind send or receive sockets when using "
        "intermediate device for request messages.")
  Process(target = _LaunchRequestStreamer, args = (config,)).start()
  if config.result_sender.url == config.result_receiver.url:
    raise ConfigException("Must set different URLs for send and receive "
        "sockets to use intermediate device for result messages.")
  if config.result_sender.bind or config.result_receiver.bind:
    raise ConfigException("Must not bind send or receive sockets when using "
        "intermediate device for result messages.")
  _LaunchResultStreamer(config)

def LaunchWorker(config, num_processes = None):
  """Start a cluster worker on the local host.
  RETURN This method does not return unless the worker receives a QUIT
  command."""
  # TODO make worker more robust to errors (e.g., trying to imprint on an image
  # that worker can't find will raise an error in the client, but also leave
  # worker in a non-recoverable state).
  if num_processes != None:
    num_processes = int(num_processes)
  pool = multiprocessing.Pool(num_processes)

  def worker_callback(dynamic_batch_request):
    """Convert a dynamic batch request to a batch result."""
    function, batch_request = dynamic_batch_request
    logging.info("glimpse_cluster.worker_callback: mapping function across "
        "%d elements" % len(batch_request))
    result_list = pool.map(function, batch_request)
    return result_list  # return batch result

  # If set, the value is the length of time without requests before the worker
  # quits.
  receiver_timeout = None
  worker = zmq_cluster.Worker(zmq.Context(), config.request_receiver,
      config.result_sender, worker_callback,
      command_receiver = config.command_receiver,
      receiver_timeout = receiver_timeout)
  worker.Setup()  # connect/bind sockets and prepare for work
  worker.Run()  # run the request/reply loop until termination
