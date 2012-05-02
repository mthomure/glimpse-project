.. _command-glab:

#############
Command: glab
#############

.. program:: glab

The `glab` command can be used to perform basic Glimpse experiments without
any programming. Given a corpus of images, the glab command will

* randomly choose (non-overlapping) training and testing sets,
* extract feature vectors using the chosen hierarchical model, and
* perform object recognition using a linear SVM.

Arguments
---------

-b, --balance                   If the distribution of images per object class
                                is unequal, use a (random) sample of the larger
                                classes to correct this.
-c, --corpus=DIR                Use corpus directory DIR.
-C, --corpus-subdir=DIR         Specify a different corpus subdirectory per
                                obect class (using :option:`-C` repeatedly).
--cluster-config=FILE           Read cluster configuration from FILE
--compute-features              Compute feature vectors (implied by
                                :option:`-s`).
-e, --edit-options              Edit model options with a GUI.
-l, --layer=LAYR                Compute feature vectors from LAYR activity.
-m, --model=MODL                Use model named MODL (one of ``ml`` or ``viz2``.
                                default: ``viz2``). See :ref:`models`.
-n, --num-prototypes=NUM        Generate NUM S2 prototypes.
-o, --options=FILE              Read model options from FILE.
-p, --prototype-algorithm=ALG   Generate S2 prototypes according to algorithm
                                ALG (one of ``imprint``, ``uniform``,
                                ``shuffle``, ``histogram``, or ``normal``). See
                                :ref:`filter learning`.
-P, --prototypes=FILE           Read S2 prototypes from FILE (overrides
                                :option:`-p`).
-r, --results=FILE              Store results to FILE.
-R, --resize=NUM                Resize the minimum dimension of images to NUM
                                pixels.
-s, --svm                       Train and test an SVM classifier.
--svm-decision-values           Print the pre-thresholded SVM decision values
                                for each test image (implies -vs).
--svm-predicted-labels          Print the predicted labels for each test image
                                (implies -vs).
-t, --pool-type=TYPE            Choose the parallelization strategy (one of
                                ``multicore``, ``singlecore``, or ``cluster``.
                                default: ``multicore``). See :ref:`worker
                                pools`.
-v, --verbose                   Enable verbose logging.
-x, --cross-validate            Compute test accuracy via cross-validation.

Corpus Format
-------------

The corpus directory is expected to contain image files grouped by object
class into sub-directories. Image files and sub-directories can have
arbitrary names. An example corpus of animals might have the following
structure. ::

   animals/
      cats/
         persian.jpg
         siamese.jpg
         tabby.jpg
      dogs/
         labrador.jpg
         poodle.jpg
         greyhound.jpg

In this case, the corpus contains a ``cats`` class and a ``dogs`` class, where
each class contains three images.

Example Usage
-------------

Use imprinted prototypes with a corpus directory named ``animals``,
saving results to ``results.dat``. ::

   > glab -s -r results.dat -c animals/ -p imprint

As above, but use 25 random prototypes and present a graphical UI to edit the
model options. ::

   > glab -s -r results.dat -c animals/ -p uniform -n 25 -e

As above, but instead read prototypes from the file ``protos.dat``,
and specify corpus directories for each object class. ::

   > glab -s -r results.dat -P protos.dat -C animals/cats -C animals/dogs

As in first example, but construct feature vectors from S1 activity,
use the ``ml`` model, and read options from ``model-options.dat``. ::

   > glab -s -r results.dat -c animals/ -l S1 -m ml -o model-options.dat

As in first example, but use a compute cluster.
Cluster information is read from the file ``cluster.ini``. ::

   > glab -s -r results.dat -c animals/ -p imprint -t cluster \
          --cluster-config=cluster.ini

Use random S2 prototypes and print the prethresholded SVM decision values.
(Note that decision values and accuracies are purely illustrative.) ::

   > glab -c animals -p uniform --svm-decision-values

   Making 10 uniform random prototypes
     done: 0.000919103622437 s
   Train SVM on 2 images
     and testing on 4 images
     done: 0.0561721324921 s
   Time to compute feature vectors: 1.4273121357 s
   Train Accuracy: 1.000
   Test Accuracy: 0.500
   Decision Values:
   cats/persian.jpg 3.31928838004e-93
   dogs/labrador.jpg 4.27620813638e-22
