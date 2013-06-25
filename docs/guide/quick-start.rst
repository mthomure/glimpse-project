###########
Quick Start
###########

To get started quickly with Glimpse, use the :ref:`glab` API. As an example,
we perform object detection on a sample dataset using an HMAX-like model.::

  >>> from glimpse.glab.api import *

  >>> Verbose()

  >>> SetSampleCorpus('easy')
  INFO:root:Reading class sub-directories from: sample-corpora/easy
  INFO:root:Reading images from class directories:
  ['sample-corpora/easy/circle',
   'sample-corpora/easy/cross']

  >>> ImprintS2Prototypes(10)
  INFO:root:Using pool: MulticorePool
  INFO:root:Learning 10 prototypes at 1 sizes from 4 images by imprinting
  Time: 0:00:01   |##############################|   Speed:  3.04  unit/s
  INFO:root:Learning prototypes took 1.318s

  >>> EvaluateClassifier()
  INFO:root:Computing C2 activation maps for 10 images
  Time: 0:00:01   |##############################|   Speed:  5.56  unit/s
  INFO:root:Computing activation maps took 1.798s
  INFO:root:Evaluating classifier on fixed train/test split on 10 images
            using 10 features from layer(s): C2
  INFO:root:Training on 4 images took 0.002s
  INFO:root:Classifier is Pipeline(learner=LinearSVC([...OUTPUT REMOVED...]))
  INFO:root:Classifier accuracy on training set is 1.000000
  INFO:root:Scoring on training set (4 images) took 0.001s
  INFO:root:Scoring on testing set (6 images) took 0.000s
  INFO:root:Classifier accuracy on test set is 1.000000

