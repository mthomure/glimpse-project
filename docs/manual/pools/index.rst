.. _worker pools:

############
Worker Pools
############

.. toctree::
   :hidden:

   gearman_pool


.. currentmodule:: glimpse.pools

A worker pool implements a strategy for parallelizing a :func:`map` operation.
That is, given a set of elements and a function taking a single input, a worker
pool returns the result of evaluating that function on each input element in
turn. In Glimpse, the function is usually a model's BuildLayer method, and the
elements are generally the model's input state for different images.

When not using a compute cluster, the best worker pool to use is generally
the one returned by :func:`MakePool`. For example

   >>> pool = glimpse.pools.MakePool()
   >>> pool.map(hex, [1, 2, 3])
   ['0x1', '0x2', '0x3']

Single-Host Pools
------------------------

The most common parallelization scheme is the :class:`MulticorePool`, which
spreads evaluation of elements across multiple cores of a single host.
Additionally, a fall-back scheme is provided by the :class:`SinglecorePool`,
which uses the builtin :func:`map` function. This can be useful for
debugging, and when the complexity of multicore communication is unwanted.

.. caution::

   Not all functions can be used in a parallel fashion. In case of
   mystifying errors, check the documentation for :class:`MulticorePool`.
   Additionally, try using :class:`SinglecorePool` to identify whether the
   error is due to parallelization.

Multi-Host Pools
-----------------------

Multi-host worker pools, or *cluster pools*, are more advanced than
single-host pools, and require some additional configuration. These
algorithms spread work across available cores on multiple machines connected
over the network. The most stable cluster pools is the :ref:`Gearman cluster
<gearman cluster>`, which can be accessed as

   >>> config_file = "config.ini"
   >>> pool = glimpse.pools.gearman_pool.MakePool(config_file)
   >>> pool.map(hex, [1, 2, 3])
   ['0x1', '0x2', '0x3']

where the file ``config.ini`` contains network configuration information for
the cluster. Cluster pools can also be accessed by name using
:func:`GetClusterPackage`.

   >>> pkg = glimpse.pools.GetClusterPackage("gearman")
   >>> pool = pkg.MakePool(config_file)

.. note::

   When implementing a new cluster pool package, be sure to include the
   ``MakePool()`` and ``RunMain()`` function. The first should return a new
   client connection, while the second should provide a command-line
   interface for the cluster.
