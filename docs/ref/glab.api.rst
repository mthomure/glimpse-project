glimpse.glab.api
================

.. module:: glimpse.glab.api

.. data:: DEFAULT_LAYER

   Default model layer to use for evaluation.

.. autofunction:: Reset
.. autofunction:: SetParams
.. autofunction:: SetParamsWithGui

   This presents a display similar to that shown in :ref:`Figure 1 <param-editor>`.

.. _param-editor:
.. figure:: ../_static/glimpse-param-editor.png
   :scale: 50%
   :align: center

   Figure 1: A screenshot of the editor for model parameters.

.. autofunction:: LoadParams
.. autofunction:: SetLayer
.. autofunction:: StoreExperiment
.. autofunction:: LoadExperiment
.. autofunction:: GetExperiment
.. autofunction:: Verbose
.. autofunction:: GetModel
.. autofunction:: SetCorpus
.. autofunction:: SetCorpusSubdirs
.. autofunction:: SetCorpusSplit
.. autofunction:: SetCorpusByName
.. autofunction:: SetS2Prototypes
.. autofunction:: ImprintS2Prototypes
.. autofunction:: MakeUniformRandomS2Prototypes
.. autofunction:: MakeShuffledRandomS2Prototypes
.. autofunction:: MakeHistogramRandomS2Prototypes
.. autofunction:: MakeNormalRandomS2Prototypes
.. autofunction:: MakeKmeansS2Prototypes
.. autofunction:: ComputeActivation
.. autofunction:: EvaluateClassifier
.. autofunction:: GetFeatures
.. autofunction:: GetImagePaths
.. autofunction:: GetLabelNames
.. autofunction:: GetParams
.. autofunction:: GetNumPrototypes
.. autofunction:: GetPrototype
.. autofunction:: GetImprintLocation
.. autofunction:: GetEvaluationLayers
.. autofunction:: GetEvaluationResults
.. autofunction:: GetPredictions
.. autofunction:: ShowS2Activity
.. autofunction:: ShowPrototype
.. autofunction:: AnnotateImprintedPrototype
.. autofunction:: AnnotateS2Activity
.. autofunction:: AnnotateC1Activity
.. autofunction:: AnnotateS1Activity
.. autofunction:: ShowS1Kernels
