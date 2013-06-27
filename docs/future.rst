Future Work
###########

There are a number of clear ways to improve Glimpse in the future.

1. The accessibility of the project could be greatly enhanced by integrating
   Glimpse into a general machine learning framework. Ideally, this
   framework would provide a graphical interface for designing and running
   experiments. A good candidate for such a framework is the `Orange
   project`_.

2. It would be helpful to have more advanced backends, particularly one
   targeting GPUs. This could probably be written using `PyCUDA`_ or
   `Theano`_. Some code for this exists in old versions of the project, and
   should be dusted off.

3. The biggest boost to accessibility of the project could come from a
   graphical user interface, which allows the user to specify arbitrary
   network topologies. This might be done by hacking an interface out of the
   Orange project's workbench code.

.. _Orange project: http://orange.biolab.si/
.. _PyCUDA: http://mathema.tician.de/software/pycuda
.. _Theano: http://deeplearning.net/software/theano/
