Quick Start Guide
=================

Installation
------------

Using `pip`:

.. code-block:: sh

   $ pip install glimpse ipython matplotlib

.. note::

  On Mac OSX, you may need to build for a 32-bit architecture. For example,
  this happens when using 32-bit Python on a 64-bit machine. To do this,
  download and unpack the project, and then use the modified install command:

  .. code-block:: sh

     $ ARCHFLAGS='-arch i386' python setup.py install


Usage
-----

To get started quickly with Glimpse, use the :ref:`glab <glab>` API from the
ipython shell. In the example below, we perform object detection on a sample
dataset using an HMAX-like model.

.. code-block:: sh

  $ ipython --pylab

::

  >>> from glimpse.glab.api import *

  >>> SetCorpusByName("easy")

  >>> ImprintS2Prototypes(10)

  >>> EvaluateClassifier()

  >>> results = GetEvaluationResults()

  >>> print "Classification accuracy:", results.score
  0.75

  >>> StoreExperiment("my-experiment.dat")


The same experiment can be run from the command-line using the :mod:`glab
<glimpse.glab.cli>` script.

.. code-block:: sh

    $ glab -v --corpus-name easy -n 10 -p imprint -E -o my-experiment.dat
    INFO:root:Reading class sub-directories from: corpora/data/easy
    INFO:root:Reading images from class directories: ['corpora/data/easy/circle', '/corpora/data/easy/cross']
    INFO:root:Using pool: MulticorePool
    INFO:root:Learning 10 prototypes at 1 sizes from 4 images by imprinting
    Time: 0:00:01   |#######################################|   Speed:  3.00  unit/s
    INFO:root:Learning prototypes took 1.334s
    INFO:root:Computing C2 activation maps for 10 images
    Time: 0:00:01   |#######################################|   Speed:  5.57  unit/s
    INFO:root:Computing activation maps took 1.795s
    INFO:root:Evaluating classifier on fixed train/test split on 10 images using 10 features from layer(s): C2
    INFO:root:Training on 4 images took 0.003s
    INFO:root:Classifier is Pipeline(learner=LinearSVC([...OUTPUT REMOVED...]))
    INFO:root:Classifier accuracy on training set is 1.000000
    INFO:root:Scoring on training set (4 images) took 0.001s
    INFO:root:Scoring on testing set (6 images) took 0.000s
    INFO:root:Classifier accuracy on test set is 1.000000

.. note::

   If you have trouble getting access to the `glab` command, check the
   :ref:`note about system paths <glab-path-note>`.

