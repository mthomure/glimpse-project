.. _command-glimpse-cluster:

########################
Command: glimpse-cluster
########################

The `glimpse-cluster` command provides the ability to

* launch worker nodes, and
* get the status for running worker nodes.

.. program:: glimpse-cluster

Usage of the command is::

   glimpse-cluster [options] CMD [ARGS]

Supported values for *options* and `CMD` depend on the cluster implementation,
which is chosen by setting the :envvar:`GLIMPSE_CLUSTER_TYPE` environment
variable. Usage for each implementation are shown below.

Gearman Cluster Usage
---------------------

For the `gearman` cluster, supported *options* include:

-c FILE   Read socket configuration from FILE.
-v        Be verbose with logging.

Supported values for the *CMD* argument include:

LaunchBroker
   Start a set of intermediate devices for the cluster on this machine.
LaunchWorker
   Start a cluster worker on the local host.
KillWorkers
   Send a *quit* command to any workers running on the cluster.
RestartWorkers
   Send *restart* to any workers running on cluster, causing a restart.
PingWorkers
   Determine the set of active workers.

IPython Cluster Usage
---------------------

For the `ipython` cluster, supported *options* include:

-p PROF   Set the IPython profile.
-v        Be verbose with logging.

and *CMD* can be one of:

KillWorkers
   Kill all active workers.
RestartWorkers
   Kill and relaunch all active workers.
PingWorkers
   Determine the set of active workers.

ZeroMQ Cluster Usage
--------------------

For the `zmq` cluster, supported *options* include:

-c FILE   Read socket configuration from FILE.
-v        Be verbose with logging.

and *CMD* can be one of:

LaunchBrokers
   Start a set of intermediate devices for the cluster on this machine.
LaunchWorker
   Start a cluster worker on the local host.
KillWorkers
   Send a *quit* command to any workers running on the cluster.
RestartWorkers
   Send *restart* to any workers running on cluster, causing a restart.
PingWorkers
   Determine the set of active workers.
FlushCluster
   Consume any queued messages waiting on the cluster.
