
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

cimport numpy as np
import numpy as np
import operator

# Datatype used to encode neural activation.
ctypedef np.float32_t act_t
activation_dtype = np.float32

# Datatype used to encode a bitset.
ctypedef unsigned char byte_t

cdef extern from "c_src/array.h":
  cdef cppclass ArrayRef2D[T]:
    ArrayRef2D(T* data, int height, int width)
    T& operator()(int y, int x)
    int w0, w1
  cdef cppclass ArrayRef3D[T]:
    ArrayRef3D(T* data, int num_bands, int height, int width)
  cdef cppclass ArrayRef4D[T]:
    ArrayRef4D(T* data, int num_out_bands, int num_in_bands, int height,
        int width)

def __CheckNumpyArrayForWrap(np.ndarray in_data not None):
  assert in_data.flags['C_CONTIGUOUS'], "Can only wrap arrays with contiguous data in row-major order."
  # XXX in the future, we'd like to support read-only access to non-writable numpy arrays.
  assert in_data.flags['WRITEABLE'], "Can only interact with non-writable arrays using const wrapper"

# TODO add ability to allocate SSE-compatible arrays.
cdef class WrapArray2D:
  cdef ArrayRef2D[act_t]* ptr
  def __cinit__(self, np.ndarray[act_t, ndim = 2] in_data not None):
    self.ptr = NULL
    __CheckNumpyArrayForWrap(in_data)
    self.ptr = new ArrayRef2D[act_t](<act_t*>in_data.data, in_data.shape[0],
        in_data.shape[1])
  def __dealloc__(self):
    if self.ptr != NULL:
      del self.ptr

cdef class WrapArray3D:
  cdef ArrayRef3D[act_t]* ptr
  def __cinit__(self, np.ndarray[act_t, ndim = 3] in_data not None):
    self.ptr = NULL
    __CheckNumpyArrayForWrap(in_data)
    self.ptr = new ArrayRef3D[act_t](<act_t*>in_data.data, in_data.shape[0],
        in_data.shape[1], in_data.shape[2])
  def __dealloc__(self):
    if self.ptr != NULL:
      del self.ptr

cdef class WrapArray4D:
  cdef ArrayRef4D[act_t]* ptr
  def __cinit__(self, np.ndarray[act_t, ndim = 4] in_data not None):
    self.ptr = NULL
    __CheckNumpyArrayForWrap(in_data)
    self.ptr = new ArrayRef4D[act_t](<act_t*>in_data.data, in_data.shape[0],
        in_data.shape[1], in_data.shape[2], in_data.shape[3])
  def __dealloc__(self):
    if self.ptr != NULL:
      del self.ptr


cdef extern from "c_src/util.h":
  bint IsEnabledSSE()
  bint IsEnabledDebugging()
  void CMaxOutputDimensions(int kheight, int kwidth, int scaling,
      int input_height, int input_width, int* output_height, int* output_width) \
      except +
  void CMaxOutputDimensionsSSE(int kheight, int kwidth, int scaling,
      int input_height, int input_width, int* output_height, int* output_width,
      int* multi_kwidth, int* lpad) except +
  void CInputDimensionsForOutput(int kheight, int kwidth, int scaling,
      int output_height, int output_width, int* input_height, \
      int* input_width) except +
  void CInputDimensionsForOutputSSE(int kheight, int kwidth, int scaling,
      int output_height, int output_width, int* input_height,
      int* input_width) except +

__USE_SSE = IsEnabledSSE()

def SetUseSSE(bint flag):
  global __USE_SSE
  if flag:
    if IsEnabledSSE():
      __USE_SSE = True
    else:
      import sys
      sys.stderr.write("WARN: SSE intrinsics not available\n")
  else:
    __USE_SSE = False

def GetUseSSE():
  global __USE_SSE
  return __USE_SSE

def GetUseDebugging():
  return IsEnabledDebugging()

# TODO check if we can express iterable type on in_shape as constraint
def MaxOutputDimensions(int kheight, int kwidth, int scaling, tuple in_shape,
    bint use_sse = None):
  cdef int oheight, owidth
  cdef int multi_kwidth, left_pad
  cdef int iheight = int(in_shape[0])
  cdef int iwidth = in_shape[1]
  if (use_sse == None and __USE_SSE) or use_sse:
    CMaxOutputDimensionsSSE(kheight, kwidth, scaling, iheight, iwidth, &oheight,
        &owidth, &multi_kwidth, &left_pad)
  else:
    CMaxOutputDimensions(kheight, kwidth, scaling, iheight, iwidth, &oheight,
        &owidth)
    lpad = 0
  return oheight, owidth, multi_kwidth, left_pad

def InputDimensionsForOutput(int kheight, int kwidth, int scaling, int oheight,
    int owidth, bint use_sse):
  """Get the size of the input layer that corresponds to the given output layer
  size."""
  cdef int iheight, iwidth
  if use_sse:
    CInputDimensionsForOutputSSE(kheight, kwidth, scaling, oheight, owidth,
        &iheight, &iwidth)
  else:
    CInputDimensionsForOutput(kheight, kwidth, scaling, oheight, owidth,
        &iheight, &iwidth)
  return iheight, iwidth

cdef extern from "c_src/array.h":
  void CNormalizeArrayAcrossBand_UnitNorm(ArrayRef3D[float]* data)

def NormalizeArrayAcrossBand_UnitNorm(np.ndarray[act_t, ndim = 3] data not
    None):
  """Scale each location of the 3D array to have unit norm."""
  data_ref = WrapArray3D(data);
  CNormalizeArrayAcrossBand_UnitNorm(data_ref.ptr)


########## Retinal Layer Processing ################

cdef extern from "c_src/retinal_layer.h" nogil:
  void CContrastEnhance(ArrayRef2D[float]& idata, int kheight, int kwidth,
      float bias, ArrayRef2D[float]& odata) except +
  void CProcessRetina(ArrayRef2D[float]& idata, int kheight, int kwidth,
      float bias, ArrayRef2D[float]& odata) except +
  void CProcessRetinaSSE(ArrayRef2D[float]& idata, int kheight, int kwidth,
      float bias, ArrayRef2D[float]& odata) except +

def ContrastEnhance(np.ndarray[act_t, ndim = 2] in_data not None, int kheight,
    int kwidth, float bias, np.ndarray[act_t, ndim = 2] out_data = None):
  """Apply retinal processing to the given image data.
  kwidth - width of retinal kernel
  bias - constant added to the stdev of local activity (avoids amplifying
         noise)"""
  if out_data == None:
    if kheight != kwidth:
      raise ValueError("Kernel must be square (spatially)")
    oheight, owidth, multi_kwidth, lpad = MaxOutputDimensions(kheight, kwidth,
        1, (in_data.shape[0], in_data.shape[1]), use_sse = False)
    if oheight <= 0 or owidth <= 0:
      raise InsufficientSizeException()
    out_data = np.empty((oheight, owidth), activation_dtype)
  i = WrapArray2D(in_data)
  o = WrapArray2D(out_data)
  CContrastEnhance(i.ptr[0], kheight, kwidth, bias, o.ptr[0])
  return out_data

def ProcessRetina(np.ndarray[act_t, ndim = 2] in_data not None, int kheight,
    int kwidth, float bias, np.ndarray[act_t, ndim = 2] out_data not None):
  """Apply retinal processing to the given image data.
  kwidth - width of retinal kernel
  bias - constant added to the stdev of local activity (avoids amplifying
         noise)"""
  in_data_ref = WrapArray2D(in_data)
  out_data_ref = WrapArray2D(out_data)
  if __USE_SSE:
    CProcessRetinaSSE(in_data_ref.ptr[0], kheight, kwidth, bias,
        out_data_ref.ptr[0])
  else:
    CProcessRetina(in_data_ref.ptr[0], kheight, kwidth, bias,
        out_data_ref.ptr[0])


########## Simple Layer Processing ################

cdef extern from "c_src/simple_layer.h" nogil:
  void CProcessSimpleLayer(ArrayRef3D[float]& in_data,
      ArrayRef4D[float]& kernels, float bias, float beta, int scaling,
      ArrayRef3D[float]& out_data)
  void CProcessSimpleLayerSSE_NoScaling(ArrayRef3D[float]& in_data,
      ArrayRef4D[float]& kernels, float bias, float beta,
      ArrayRef3D[float]& output)

  void CDotProduct(ArrayRef3D[float]& in_data, ArrayRef4D[float]& kernels,
      int scaling, ArrayRef3D[float]& out_data)
  void CNormDotProduct(ArrayRef3D[float]& in_data, ArrayRef4D[float]& kernels,
      float bias, int scaling, ArrayRef3D[float]& out_data)
  void CNormRbf(ArrayRef3D[float]& in_data, ArrayRef4D[float]& kernels,
      float bias, float beta, int scaling, ArrayRef3D[float]& out_data)
  void CRbf(ArrayRef3D[float]& in_data, ArrayRef4D[float]& kernels,
      float beta, int scaling, ArrayRef3D[float]& out_data)

def ProcessSimpleLayer(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None, float bias, float beta,
    int scaling, np.ndarray[act_t, ndim=3] out_data not None, bint use_sse):
  assert in_data.shape[0] == kernels.shape[1], "Number of bands in kernel " \
      "must match number of bands in input data"
  assert kernels.shape[0] == out_data.shape[0], "Number of kernels must match" \
      " number of bands in output array"
  in_data_ref = WrapArray3D(in_data)
  kernels_ref = WrapArray4D(kernels)
  out_data_ref = WrapArray3D(out_data)
  if __USE_SSE and use_sse:
    CProcessSimpleLayerSSE_NoScaling(in_data_ref.ptr[0], kernels_ref.ptr[0],
        bias, beta, out_data_ref.ptr[0])
  else:
    CProcessSimpleLayer(in_data_ref.ptr[0], kernels_ref.ptr[0], bias, beta,
        scaling, out_data_ref.ptr[0])

class InsufficientSizeException(BaseException):
  """Exception indicating that the input array was too small (spatially) to
  support the requested operation."""
  pass

def _PrepareFilterArgs(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None,
    np.ndarray[act_t, ndim=3] out_data = None, scaling = 1):
  if out_data == None:
    kheight = kernels.shape[2]
    kwidth = kernels.shape[3]
    if kheight != kwidth:
      raise ValueError("Kernel must be square (spatially)")
    oheight, owidth, multi_kwidth, lpad = MaxOutputDimensions(kheight, kwidth, scaling,
        (in_data.shape[1], in_data.shape[2]), use_sse = False)
    nkernels = kernels.shape[0]
    if oheight <= 0 or owidth <= 0:
      raise InsufficientSizeException()
    out_data = np.empty((nkernels, oheight, owidth), np.float32)
  if in_data.shape[0] != kernels.shape[1]:
    raise ValueError("Number of bands in kernel must match number of bands in input data")
  if kernels.shape[0] != out_data.shape[0]:
    raise ValueError("Number of kernels must match number of bands in output array")
  return out_data #in_data, kernels, out_data

def DotProduct(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None,
    np.ndarray[act_t, ndim=3] out_data = None, int scaling = 1):
  out_data = _PrepareFilterArgs(in_data, kernels, out_data, scaling)
  i = WrapArray3D(in_data)
  k = WrapArray4D(kernels)
  o = WrapArray3D(out_data)
  CDotProduct(i.ptr[0], k.ptr[0], scaling, o.ptr[0])
  return out_data

def NormDotProduct(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None,
    np.ndarray[act_t, ndim=3] out_data = None, float bias = 1.0,
    int scaling = 1):
  out_data = _PrepareFilterArgs(in_data, kernels, out_data, scaling)
  i = WrapArray3D(in_data)
  k = WrapArray4D(kernels)
  o = WrapArray3D(out_data)
  CNormDotProduct(i.ptr[0], k.ptr[0], bias, scaling, o.ptr[0])
  return out_data

def NormRbf(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None,
    np.ndarray[act_t, ndim=3] out_data = None, float bias = 1, float beta = 1,
    int scaling = 1):
  out_data = _PrepareFilterArgs(in_data, kernels, out_data, scaling)
  i = WrapArray3D(in_data)
  k = WrapArray4D(kernels)
  o = WrapArray3D(out_data)
  CNormRbf(i.ptr[0], k.ptr[0], bias, beta, scaling, o.ptr[0])
  return out_data

def Rbf(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None,
    np.ndarray[act_t, ndim=3] out_data = None, float beta = 1,
    int scaling = 1):
  out_data = _PrepareFilterArgs(in_data, kernels, out_data, scaling)
  i = WrapArray3D(in_data)
  k = WrapArray4D(kernels)
  o = WrapArray3D(out_data)
  CRbf(i.ptr[0], k.ptr[0], beta, scaling, o.ptr[0])
  return out_data

########## Complex Layer Processing ################

cdef extern from "c_src/bitset_array.h":
  cdef cppclass BitsetArrayRef:
    BitsetArrayRef(byte_t* data, int num_sets, int num_bits_per_set)
  cdef cppclass BitsetArrayRef3D:
    BitsetArrayRef3D(BitsetArrayRef bar, int num_bands, int height, int width)

cdef class WrapBitsetArray3D:
  cdef BitsetArrayRef3D* ptr
  def __cinit__(self, np.ndarray[byte_t, ndim = 4] in_data not None):
    num_sets = in_data.shape[0] * in_data.shape[1] * in_data.shape[2]
    num_bits_per_set = in_data.shape[3] * 8
    self.ptr = new BitsetArrayRef3D(
        BitsetArrayRef(<byte_t*>in_data.data, num_sets, num_bits_per_set),
        in_data.shape[0], in_data.shape[1], in_data.shape[2])
  def __dealloc__(self):
    del self.ptr

cdef extern from "c_src/complex_layer.h" nogil:
  void CProcessC1Layer_PoolSpaceAndPhase(ArrayRef4D[float]& input,
      int kheight, int kwidth, int scaling, ArrayRef3D[float]& output,
      BitsetArrayRef3D& max_coords) except +
  void CProcessComplexLayer(ArrayRef3D[float]& input, int kheight,
      int kwidth, int scaling, ArrayRef3D[float]& output,
      BitsetArrayRef3D& max_coords) except +

def ProcessC1Layer_PoolSpaceAndPhase(
    np.ndarray[act_t, ndim = 4] in_data not None,
    int kheight, int kwidth, int scaling,
    np.ndarray[act_t, ndim = 3] out_data not None,
    np.ndarray[byte_t, ndim = 4] max_coords not None):
  in_data_ref = WrapArray4D(in_data)
  out_data_ref = WrapArray3D(out_data)
  max_coords_ref = WrapBitsetArray3D(max_coords)
  CProcessC1Layer_PoolSpaceAndPhase(in_data_ref.ptr[0], kheight, kwidth,
      scaling, out_data_ref.ptr[0], max_coords_ref.ptr[0])

def ProcessComplexLayer(np.ndarray[act_t, ndim = 3] idata not None,
    int kheight, int kwidth, int scaling,
    np.ndarray[act_t, ndim = 3] odata not None,
    np.ndarray[byte_t, ndim = 4] max_coords not None):
  idata_ref = WrapArray3D(idata)
  odata_ref = WrapArray3D(odata)
  max_coords_ref = WrapBitsetArray3D(max_coords)
  CProcessComplexLayer(idata_ref.ptr[0], kheight, kwidth,
      scaling, odata_ref.ptr[0], max_coords_ref.ptr[0])

