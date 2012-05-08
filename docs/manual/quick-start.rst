###########
Quick Start
###########

To get started quickly with Glimpse, use the :ref:`glab` API. As an example,
we perform object detection on the AnimalDB dataset used by Serre et al
[1]_, using an HMAX-like model. (This was run on a 2.33 GHz Intel Core2 Quad
CPU.) ::

   >>> from glimpse.glab import *
   >>> Verbose()
   >>> image_dir = "/path/to/AnimalDB"
   >>> SetCorpus(image_dir)
   >>> ImprintS2Prototypes(100)
   Imprinting 100 prototypes
     done: 18.6740250587 s
   >>> RunSvm()
   Train SVM on 600 images
     and testing on 600 images
     done: 0.471097946167 s
   Time to compute feature vectors: 322.276679039 s
   Accuracy is 0.917 on training set, and 0.792 on test set.

References
----------

.. [1] ﻿Serre, T., Oliva, A. & Poggio, T., 2007. A feedforward architecture
   accounts for rapid categorization. Proceedings of the National Academy of
   Sciences, 104(15), p.6424-6429.
