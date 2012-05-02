###########
Future Work
###########

There are a number of clear ways to improve Glimpse in the future. Here is
the short list.

First, the accessibility of the project could be greatly enhanced by
integrating Glimpse into a general machine learning framework. Ideally, this
framework would provide a graphical interface for designing and running
experiments. (This would largely replace the :ref:`glab` interface.) A good
candidate for such a framwork is the `Orange project`_.

.. _Orange project: http://orange.biolab.si/

Second, it would be helpful to have more advanced backends, particularly one
targeting GPUs. This could probably be written using `PyCUDA`_ or `Theano`_.

.. _PyCUDA: http://mathema.tician.de/software/pycuda
.. _Theano: http://deeplearning.net/software/theano/

Finally, the biggest boost to accessibility of the project could come from a
graphical user interface, which allows the user to specify arbitrary network
topologies. This might be done by hacking an interface out of the Orange
project's workbench code.
