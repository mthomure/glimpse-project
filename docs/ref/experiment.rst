glimpse.experiment
==================

.. automodule:: glimpse.experiment


Data Structures
---------------

.. autoclass:: CorpusData
.. autoclass:: ExtractorData
.. autoclass:: EvaluationData
.. autoclass:: ExperimentData


Running Experiments
--------------------

.. autofunction:: Verbose
.. autofunction:: SetModel
.. autofunction:: SetCorpus
.. autofunction:: SetCorpusSubdirs
.. autofunction:: SetCorpusSplit
.. autofunction:: MakePrototypes
.. autofunction:: ComputeActivation
.. autofunction:: TrainAndTestClassifier
.. autofunction:: CrossValidateClassifier
.. autofunction:: ExtractFeatures
.. autofunction:: ExtractHistogramFeatures


Prototype Algorithms
--------------------

.. autoclass:: ImprintAlg
.. autoclass:: ShuffledAlg
.. autoclass:: UniformAlg
.. autoclass:: KmeansAlg
   :no-members:
.. autoclass:: MFWKmeansAlg
.. autoclass:: OMWKmeansAlg
.. autofunction:: GetAlgorithmNames


Analyzing Results
-------------------------

.. autofunction:: GetImagePaths
.. autofunction:: GetLabelNames
.. autofunction:: GetParams
.. autofunction:: GetNumPrototypes
.. autofunction:: GetImprintLocation
.. autofunction:: GetPrototype
.. autofunction:: GetImagePatchForImprintedPrototype
.. autofunction:: GetBestPrototypeMatch
.. autofunction:: GetImageActivity
.. autofunction:: GetTrainingSet
.. autofunction:: GetPredictions
.. autofunction:: GetEvaluationLayers
.. autofunction:: GetEvaluationResults
.. autofunction:: ShowS2Activity
.. autofunction:: AnnotateS2Activity
.. autofunction:: AnnotateBestPrototypeMatch
.. autofunction:: AnnotateImprintedPrototype
.. autofunction:: ShowPrototype
.. autofunction:: ShowC1Activity
.. autofunction:: AnnotateC1Activity
.. autofunction:: ShowS1Activity
.. autofunction:: AnnotateS1Activity
.. autofunction:: ShowS1Kernels


Miscellaneous
-------------

.. autoexception:: ExpError
.. autoclass:: DirReader
.. autofunction:: ReadCorpusDirs
.. autofunction:: ResolveLayers
.. autofunction:: BalanceCorpus
.. autofunction:: GetCorpusByName


Subpackages
-----------

.. toctree::

   experiment.mf_wkmeans
   experiment.om_wkmeans
