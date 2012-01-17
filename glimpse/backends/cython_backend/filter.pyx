# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

###### UTIL DECLARATIONS ##################

cimport numpy as np
from glimpse.backends.backend import InsufficientSizeException
import numpy as np
import operator

# Datatype used to encode neural activation.
ctypedef np.float32_t act_t
activation_dtype = np.float32

# Datatype used to encode a bitset.
ctypedef unsigned char byte_t

cdef extern from "src/array.h":
  cdef cppclass ArrayRef2D[T]:
    ArrayRef2D(T* data, int height, int width)
    T& operator()(int y, int x)
    int w0, w1
  cdef cppclass ArrayRef3D[T]:
    ArrayRef3D(T* data, int num_bands, int height, int width)
  cdef cppclass ArrayRef4D[T]:
    ArrayRef4D(T* data, int num_out_bands, int num_in_bands, int height,
        int width)

cdef extern from "src/bitset_array.h":
  cdef cppclass BitsetArrayRef:
    BitsetArrayRef(byte_t* data, int num_sets, int num_bits_per_set)
  cdef cppclass BitsetArrayRef3D:
    BitsetArrayRef3D(BitsetArrayRef bar, int num_bands, int height, int width)

cdef extern from "src/util.h":
  bint IsEnabledDebugging()
  void COutputMapShapeForInput(int kheight, int kwidth, int scaling,
      int input_height, int input_width, int* output_height, \
      int* output_width) except +
  void CInputMapShapeForOutput(int kheight, int kwidth, int scaling,
      int output_height, int output_width, int* input_height, \
      int* input_width) except +

###### UTIL DEFINITIONS ##################

def __CheckNumpyArrayForWrap(np.ndarray in_data not None):
  assert in_data.flags['C_CONTIGUOUS'], "Can only wrap arrays with contiguous" \
      " data in row-major order."
  # XXX in the future, we'd like to support read-only access to non-writable
  # numpy arrays.
  assert in_data.flags['WRITEABLE'], "Can only interact with non-writable" \
      " arrays using const wrapper"

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

def GetUseDebugging():
  return IsEnabledDebugging()

def OutputMapShapeForInput(int kheight, int kwidth, int scaling, int iheight,
    int iwidth):
  """Get the size of the output layer that corresponds to the given input layer
  size."""
  cdef int oheight, owidth
  COutputMapShapeForInput(kheight, kwidth, scaling, iheight, iwidth, &oheight,
      &owidth)
  return oheight, owidth

def InputMapShapeForOutput(int kheight, int kwidth, int scaling, int oheight,
    int owidth):
  """Get the size of the input layer that corresponds to the given output layer
  size."""
  cdef int iheight, iwidth
  CInputMapShapeForOutput(kheight, kwidth, scaling, oheight, owidth, &iheight,
      &iwidth)
  return iheight, iwidth

###### OPS DECLARATIONS ##################

cdef extern from "src/filter.h" nogil:
  void CContrastEnhance(ArrayRef2D[float]& idata, int kheight, int kwidth,
      float bias, ArrayRef2D[float]& odata) except +
  void CDotProduct(ArrayRef3D[float]& in_data, ArrayRef4D[float]& kernels,
      int scaling, ArrayRef3D[float]& out_data)
  void CNormDotProduct(ArrayRef3D[float]& in_data, ArrayRef4D[float]& kernels,
      float bias, int scaling, ArrayRef3D[float]& out_data)
  void CNormRbf(ArrayRef3D[float]& in_data, ArrayRef4D[float]& kernels,
      float bias, float beta, int scaling, ArrayRef3D[float]& out_data)
  void CRbf(ArrayRef3D[float]& in_data, ArrayRef4D[float]& kernels,
      float beta, int scaling, ArrayRef3D[float]& out_data)
  void CLocalMax(ArrayRef3D[float]& input, int kheight, int kwidth, int scaling,
      ArrayRef3D[float]& output) except +

###### OPS DEFINITIONS ##################

def ContrastEnhance(np.ndarray[act_t, ndim = 2] in_data not None, int kheight,
    int kwidth, float bias, np.ndarray[act_t, ndim = 2] out_data = None):
  """Apply retinal processing to the given image data.
  kwidth - width of retinal kernel
  bias - constant added to the stdev of local activity (avoids amplifying
         noise)"""
  if kheight != kwidth:
    raise ValueError("Kernel must be square (spatially)")
  if out_data == None:
    oheight, owidth = OutputMapShapeForInput(kheight, kwidth,
        1,  # scaling
        in_data.shape[0], in_data.shape[1])
    if oheight <= 0 or owidth <= 0:
      raise InsufficientSizeException
    out_data = np.empty((oheight, owidth), activation_dtype)
  i = WrapArray2D(in_data)
  o = WrapArray2D(out_data)
  CContrastEnhance(i.ptr[0], kheight, kwidth, bias, o.ptr[0])
  return out_data

def _Prepare3dFilterArgs(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None,
    np.ndarray[act_t, ndim=3] out_data = None, scaling = 1):
  if out_data == None:
    kheight = kernels.shape[2]
    kwidth = kernels.shape[3]
    if kheight != kwidth:
      raise ValueError("Kernel must be square (spatially)")
    oheight, owidth = OutputMapShapeForInput(kheight, kwidth, scaling,
        in_data.shape[1], in_data.shape[2])
    nkernels = kernels.shape[0]
    if oheight <= 0 or owidth <= 0:
      raise InsufficientSizeException
    out_data = np.empty((nkernels, oheight, owidth), np.float32)
  if in_data.shape[0] != kernels.shape[1]:
    raise ValueError("Number of bands in kernel must match number of bands in" \
        " input data")
  if kernels.shape[0] != out_data.shape[0]:
    raise ValueError("Number of kernels must match number of bands in output" \
        " array")
  return out_data

def DotProduct(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None,
    np.ndarray[act_t, ndim=3] out_data = None, int scaling = 1):
  out_data = _Prepare3dFilterArgs(in_data, kernels, out_data, scaling)
  i = WrapArray3D(in_data)
  k = WrapArray4D(kernels)
  o = WrapArray3D(out_data)
  CDotProduct(i.ptr[0], k.ptr[0], scaling, o.ptr[0])
  return out_data

def NormDotProduct(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None,
    np.ndarray[act_t, ndim=3] out_data = None, float bias = 1.0,
    int scaling = 1):
  out_data = _Prepare3dFilterArgs(in_data, kernels, out_data, scaling)
  i = WrapArray3D(in_data)
  k = WrapArray4D(kernels)
  o = WrapArray3D(out_data)
  CNormDotProduct(i.ptr[0], k.ptr[0], bias, scaling, o.ptr[0])
  return out_data

def NormRbf(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None,
    np.ndarray[act_t, ndim=3] out_data = None, float bias = 1, float beta = 1,
    int scaling = 1):
  out_data = _Prepare3dFilterArgs(in_data, kernels, out_data, scaling)
  i = WrapArray3D(in_data)
  k = WrapArray4D(kernels)
  o = WrapArray3D(out_data)
  CNormRbf(i.ptr[0], k.ptr[0], bias, beta, scaling, o.ptr[0])
  return out_data

def Rbf(np.ndarray[act_t, ndim = 3] in_data not None,
    np.ndarray[act_t, ndim=4] kernels not None,
    np.ndarray[act_t, ndim=3] out_data = None, float beta = 1,
    int scaling = 1):
  out_data = _Prepare3dFilterArgs(in_data, kernels, out_data, scaling)
  i = WrapArray3D(in_data)
  k = WrapArray4D(kernels)
  o = WrapArray3D(out_data)
  CRbf(i.ptr[0], k.ptr[0], beta, scaling, o.ptr[0])
  return out_data

def LocalMax(np.ndarray[act_t, ndim = 3] in_data not None, int kheight,
    int kwidth, np.ndarray[act_t, ndim = 3] out_data = None, int scaling = 1):
  if kheight != kwidth:
    raise ValueError("Kernel must be square (spatially)")
  if out_data == None:
    oheight, owidth = OutputMapShapeForInput(kheight, kwidth, scaling,
        in_data.shape[1], in_data.shape[2])
    if oheight <= 0 or owidth <= 0:
      raise InsufficientSizeException
    ibands = in_data.shape[0]
    out_data = np.empty((ibands, oheight, owidth), activation_dtype)
  idata_ref = WrapArray3D(in_data)
  odata_ref = WrapArray3D(out_data)
  CLocalMax(idata_ref.ptr[0], kheight, kwidth, scaling, odata_ref.ptr[0])
  return out_data
