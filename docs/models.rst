.. _models:

Models
######

In Glimpse, the model object defines the network topology, including the
number of layers and the operations used at each layer. When the object is
constructed, it is given a backend implementation and a set of parameters
that control its behavior.

The model can be viewed as a transformation between states, where a state
encodes the activity of all computed model layers. To process an image, we
first wrap the image in a new state object. The model is then applied to
transform this state to a new state, which contains activation for a higher
layer in the network. This is shown in the following example. ::

   >>> from glimpse.models.ml import BuildLayer, Model, Layer
   >>> model = Model()
   >>> istate = model.MakeState("example.jpg")
   >>> ostate = BuildLayer(model, Layer.C1, istate)
   >>> c1 = ostate[Layer.C1]

In this case, the ``c1`` variable will now contain activation for the C1 layer
of the :class:`model <glimpse.models.ml.Model>`. A feature vector can then
be derived from the activation data as:

   >>> from glimpse.util.garray import FlattenArrays
   >>> features = FlattenArrays(c1)

Oftentimes, it may be preferable to use the :mod:`glab <glimpse.glab.api>`
module. In this case, the above example could be written as::

   >>> SetLayer("C1")
   >>> features = GetImageFeatures("example.jpg")

There is currently one hierarchical model included in the Glimpse project.
It specifies an HMAX-like network, in which an alternating sequence
of "simple" and "complex" layers gradually build up object specificity while
also building invariance to certain transformations. Specifically, an image
is first :ref:`preprocessed <preprocessing>`, and then filtered with a
layer of S1 units to detect edges at various orientation and scale. The
corresponding response maps are then blurred by replacing each local
neighborhood with its maximum activation. This process is then repeated,
with a layer of S2 units being applied to result of the C1 layer. Here, each
S2 unit is characterized by the input template, or *prototype*, to which it
responds. Given N different prototypes, therefore, the S2 layer will
generate N different response maps per scale. Finally, the C2 layer
summarizes the image by computing the maximum response for each S2 prototype
for any location or scale.

To compute scale bands, the :class:`model <glimpse.models.ml.Model>` uses a
scale pyramid approach. Instead of using different S1 filters for each
scale, the model uses different-sized versions of the input image. Thus, the
course-scale response maps are computed by applying a battery of Gabors to
the original input image. Response maps for the finest-grained scale use the
same battery of Gabors, but apply them to a resized (shrunken) version of
the image.

.. _preprocessing:

Preprocessing
-------------

An initial preprocessing stage, referred to as the *retinal* layer, is used
to 1) remove color information, 2) scale the input image, and 3) enhance
image contrast. Color information is removed according to the ITU-R 601-2
luma transform (see the `Image.convert` method in the `Python Imaging
Library`_). Optionally, the input image can also be scaled (via bicubic
interpolation), such that its shorter side has a given length. Finally,
image contrast optionally is enhanced by applying the :func:`ContrastEnhance
<glimpse.backends.IBackend.ContrastEnhance>` backend function.

.. _Python Imaging Library: http://www.pythonware.com/library/pil/handbook/image.htm


.. _parameters:

Model Parameters
----------------

Behavior for each model is controlled by a set of parameters, which are
described below according to the layer they affect. To customize these
parameters, the user should first create a :class:`Params
<glimpse.models.ml.Params>` object corresponding to the model class, and then
set the desired values. An example is shown below:

   >>> from glimpse.models.ml import Params
   >>> params = Params()
   >>> params.num_scales = 8
   >>> params.s1_num_orientations = 16
   >>> m = Model(params)

Using the :mod:`glab <glimpse.glab.api>` interface, this simplifies to:

   >>> params = GetParams()
   >>> params.num_scales = 8
   >>> params.s1_num_orientations = 16

**Preprocessing Options**

Image Resizing Method
   The method to use when resize the input image. One of "score short edge",
   "scale long edge", "scale width", "scale height", "scale and crop", or
   "none". When the method is "scale and crop", use the length parameter to
   specify the output image width, and the aspect_ratio parameter to specify
   the (relative) output image height.

   >>> image_resize_method = 'scale short edge',

Image Aspect Ratio
   The aspect ratio to use when the resize method is "scale and crop".

   >>> image_resize_aspect_ratio = 1.0,

Image Length
   The output image length.

   >>> image_resize_length = 220,

Retina Enabled
   Whether to use the retinal stage during preprocessing. (Note that color
   information will always be removed.)

   >>> params.retina_enabled = False

Retina Bias
   The bias term used in the :func:`contrast enhancement <contrast enhancement>`
   function to avoid noise amplificiation.

   >>> params.retina_bias = 1.0

Retina Kernel Width
   Size of the local neighborhood used by the preprocessing function.

   >>> params.retina_kwidth = 15

**S1 and S2 Layer Options**

Beta
   Tuning parameter of the activation function (for Rbf and NormRbf).

   >>> params.s1_beta = 1.0
   >>> params.s2_beta = 5.0

Bias
   Bias term for normalization in the activation function (for NormDotProduct
   and NormRbf operations).

   >>> params.s1_bias = 0.01
   >>> params.s2_bias = 0.1

Kernel Width
   Spatial extent of the local neighborhood.

   >>> params.s1_kwidth = 11
   >>> params.s2_kwidth = [7]

.. note::

   The S2 layer supports kernels (aka prototypes) with multiple different
   widths. Thus, the `s2_kwidth` parameter is a list.

Operation
   The form of the activation function (one of DotProduct, NormDotProduct, Rbf,
   or NormRbf). See the set of :ref:`filter operations <filtering operations>`
   supported by the backends.

   >>> params.s1_operation = "NormDotProduct"
   >>> params.s2_operation = "Rbf"

Sampling
   The sub-sampling factor used when computing S-unit activation.

   >>> params.s1_sampling = 1
   >>> params.s2_sampling = 1

**S1 Gabor Filter Options**

Number of Orientations
   Number of different Gabor orientations.

   >>> params.s1_num_orientations = 4

Shift Orientations
   Whether Gabors are shifted to avoid lining up with the axes.

   >>> params.s1_shift_orientations = False

Number of Phases
   Number of different phases for the S1 Gabor filters (two phases means
   detecting a black to white transition, and vice versa).

   >>> params.s1_num_phases = 2

Number of Scales
   Number of different scales with which to analyze the image.

   >>> params.num_scales = 9

Scale Factor
   (:mod:`ml` model only) The down-sampling factor used to create course
   representations of the input image.

   >>> params.scale_factor = 2**(1/4)

**C1 and C2 Layer Options**

Kernel Width
   Size of the local neighborhood used in the C-unit pooling function.

   >>> params.c1_kwidth = 11

Sampling
   The sub-sampling factor used when computing C-unit activiation.

   >>> params.c1_sampling = 5

C1 Whiten
   Whether to whiten C1 data. See the :func:`Whiten
   <glimpse.models.ml.Whiten>` function.

   >>> params.c1_whiten = False

