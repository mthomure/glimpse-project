.. _gearman cluster:

####################
Gearman Cluster Pool
####################

.. currentmodule:: glimpse.pools.gearman_cluster

This worker pool uses a `Gearman <http://gearman.org/>`_ job server to
manage the list of tasks (i.e., function invocations), and a small number of
`ZeroMQ <http://www.zeromq.org/>`_ channels for logging and command
communication. To create a new Gearman cluster pool, use the
:func:`MakePool` function as::

   >>> config_file = "config.ini"
   >>> pool = glimpse.pools.gearman_cluster.MakePool(config_file)
   >>> pool.map(hex, [1, 2, 3])
   ['0x1', '0x2', '0x3']

Launching the Cluster
---------------------

The ``gearmand`` job server can be launched according to the `Gearman docs
<http://gearman.org/index.php?id=getting_started>`_ as::

   $ gearmand -d

and worker nodes can then be launched with the :ref:`glimpse-cluster
<command-glimpse-cluster>` command.

Config File Format
------------------

Client Section

   job_server_url
      Gearman server URL used by worker to listen for tasks, and by client to
      send tasks.
   command_backend_url
      ZeroMQ socket to which worker listens for command messages.
   log_frontend_url
      ZeroMQ socket on which worker sends logging messages.
   command_frontend_url
      ZeroMQ socket to which user sends command messages.
   log_backend_url
      ZeroMQ socket to which user listens for logging messages.

Server Section

   command_frontend_url
      ZeroMQ socket to which broker listens for command messages.
   command_backend_url
      ZeroMQ socket to which broker writes command messages.
   log_frontend_url
      ZeroMQ socket to which broker listens for logging messages.
   log_backend_url
      ZeroMQ socket to which broker writes logging messages.

A simple example of a Gearman cluster configuration is shown below. In this
example, the client communicates directly with worker nodes. ::

   [client]

      job_server_url = jobserver.com:4730
      command_backend_url = tcp://client.com:9000
      log_frontend_url = tcp://client.com:9001
      command_frontend_url = tcp://client.com:9000
      log_backend_url = tcp://client.com:9001

The following example instead uses a message broker to decouple the client
and worker nodes. This is useful, for example, if the client URL changes
over time. ::

   [client]

      job_server_url = jobserver.com:4730
      command_backend_url = tcp://broker.com:9001
      log_frontend_url = tcp://broker.com:9002
      command_frontend_url = tcp://broker.com:9000
      log_backend_url = tcp://broker.com:9003

   [server]

      command_backend_url = tcp://broker.com:9001
      command_frontend_url = tcp://broker.com:9000
      log_backend_url = tcp://broker.com:9003
      log_frontend_url = tcp://broker.com:9002
