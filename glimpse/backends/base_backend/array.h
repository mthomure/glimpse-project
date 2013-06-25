/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#ifndef __ARRAY_H__
#define __ARRAY_H__

#include "util.h"
#include <cstring>

template <class T>
class StdAllocator {
public:
  static T* AllocateArray(int size);
  static void FreeArray(T* array);
};

/** Provides an n-dimensional array wrapper around non-owned memory.
 * Supports operations of the form:
 *      wrapper[ idx ]
 *      wrapper( z, y, x )      -- i.e., for 3D array
 * Type parameters include:
 *   T:    The element type. Only used to provide default values for other
 *         template parameters.
 *   DATA: Type of underlying memory view. In the simplest case, this is just an
 *         array/pointer. The expression "data[x]" must be valid, where data is
 *         a variable of type DATA. Some constructors also require the
 *         expressions "data.Size()" and "data.Get()" to be valid.
 *   REF:  The return type of an element lookup.
 */

/** One-dimensional array wrapper. This is really only useful if you
 * want to pass array lengths around with the data, or if you want to
 * match the ArrayRefND interface for a template.
 */
template <class T, class DATA = T*, class REF = T&>
class ArrayRef1D {
 protected:
  DATA data;
  /** Called by sub-class constructors. */
  ArrayRef1D(int w0);
 public:
  const int w0;

  template <class ARRAY>
  ArrayRef1D(ARRAY& array, int w0) :
      data(array.Get()), w0(w0) {
    ASSERT_EQUALS(array.Size(), Size());
  }

  ArrayRef1D(DATA data, int w0);
  REF operator[](int x);
  const REF operator[](int x) const;
  REF operator()(int x);
  const REF operator()(int x) const;
  DATA Get();
  const DATA Get() const;
  const int Size();
};

template <class T, class A = StdAllocator<T> >
class Array1D : public ArrayRef1D<T> {
 public:
  Array1D(int w0);
  ~Array1D();
};

template <class T, class DATA=T* , class REF = T&>
class ArrayRef2D {
 protected:
  DATA data;
  /** Called by sub-class constructors. */
  ArrayRef2D(int w0, int w1);
 public:
  const int w0, w1;       // Width (i.e., extent)

  template <class ARRAY>
  ArrayRef2D(ARRAY& array, int w0, int w1) :
      data(array.Get()), w0(w0), w1(w1) {
    ASSERT_EQUALS(array.Size(), Size());
  }

  ArrayRef2D(DATA data, int w0, int w1);
  REF operator[](int x);
  REF operator()(int x0, int x1);
  const REF operator()(int x0, int x1) const;
  DATA Get();
  const DATA Get() const;
  const int Size();
};

template <class T, class A = StdAllocator<T> >
class Array2D : public ArrayRef2D<T> {
 public:
  Array2D(int w0, int w1);
  ~Array2D();
};

/** 3D array interface for a 1D linear array. */
template <class T, class DATA = T*, class REF = T&>
class ArrayRef3D {
 protected:
  DATA data;
  /** Called by sub-class constructors. */
  ArrayRef3D(int w0, int w1, int w2);
 public:
  const int w0, w1, w2;

  template <class ARRAY>
  ArrayRef3D(ARRAY& array, int w0, int w1, int w2) :
      data(array.Get()), w0(w0), w1(w1), w2(w2) {
    ASSERT_EQUALS(array.Size(), Size());
  }

  ArrayRef3D(DATA data, int w0, int w1, int w2);
  REF operator[](int x);
  REF operator()(int x0, int x1, int x2);
  const REF operator()(int x0, int x1, int x2) const;
  DATA Get();
  const DATA Get() const;
  const int Size();
};

template <class T, class A = StdAllocator<T> >
class Array3D : public ArrayRef3D<T> {
 public:
  Array3D(int w0, int w1, int w2);
  ~Array3D() ;
};

template <class T, class DATA = T*, class REF = T&>
class ArrayRef4D {
 protected:
  DATA data;
  /** Called by sub-class constructors. */
  ArrayRef4D(int w0, int w1, int w2, int w3);
 public:
  const int w0, w1, w2, w3;       // Width (i.e., extent)

  template <class ARRAY>
  ArrayRef4D(ARRAY& array, int w0, int w1, int w2, int w3) :
      data(array.Get()), w0(w0), w1(w1), w2(w2), w3(w3) {
    ASSERT_EQUALS(array.Size(), Size());
  }

  ArrayRef4D(DATA data, int w0, int w1, int w2, int w3);
  REF operator[](int x);
  REF operator()(int x0, int x1, int x2, int x3);
  const REF operator()(int x0, int x1, int x2, int x3) const;
  DATA Get();
  const DATA Get() const;
  const int Size();
};

template <class T, class A = StdAllocator<T> >
class Array4D : public ArrayRef4D<T> {
 public:
  Array4D(int w0, int w1, int w2, int w3);
  ~Array4D();
};

template <class T, class DATA = T*, class REF = T&>
class ArrayRef5D {
 protected:
  DATA data;
  /** Called by sub-class constructors. */
  ArrayRef5D(int w0, int w1, int w2, int w3, int w4);
 public:
  const int w0, w1, w2, w3, w4;   // Width (i.e., extent)

  template <class ARRAY>
  ArrayRef5D(ARRAY& array, int w0, int w1, int w2, int w3, int w4) :
      data(array.Get()), w0(w0), w1(w1), w2(w2), w3(w3), w4(w4) {
    ASSERT_EQUALS(array.Size(), Size());
  }

  ArrayRef5D(DATA data, int w0, int w1, int w2, int w3, int w4);
  REF operator[](int x);
  REF operator()(int x0, int x1, int x2, int x3, int x4);
  const REF operator()(int x0, int x1, int x2, int x3, int x4) const;
  DATA Get();
  const DATA Get() const;
  const int Size();
};

template <class T, class A = StdAllocator<T> >
class Array5D : public ArrayRef5D<T> {
 public:
  Array5D(int w0, int w1, int w2, int w3, int w4);
  ~Array5D();
};

template <class ARRAY>
void Memset(ARRAY* array, int byte_value);

/** Provides an n-dimensional array index.
 * Supports operations of the form:
 *      indexer( z, y, x )      -- i.e., for 3D array
 */
class Index2D {
 public:
  const int w0, w1;
  Index2D(int w0, int w1);
  int operator()(int x0, int x1);
};
class Index3D {
 public:
  const int w0, w1, w2;
  Index3D(int w0, int w1, int w2);
  int operator()(int x0, int x1, int x2);
};
class Index4D {
 public:
  const int w0, w1, w2, w3;
  Index4D(int w0, int w1, int w2, int w3);
  int operator()(int x0, int x1, int x2, int x3);
};
class Index5D {
 public:
  const int w0, w1, w2, w3, w4;
  Index5D(int w0, int w1, int w2, int w3, int w4);
  int operator()(int x0, int x1, int x2, int x3, int x4);
};

// Treating last two components of array as encoding (y,x) location,
// these functions allow access to the last two dimensions of array
// regardless of the array's shape.
template <class T1, class T2, class T3> int GetHeight(
    const ArrayRef2D<T1, T2, T3>& array);
template <class T1, class T2, class T3> int GetHeight(
    const ArrayRef3D<T1, T2, T3>& array);
template <class T1, class T2, class T3> int GetHeight(
    const ArrayRef4D<T1, T2, T3>& array);
template <class T1, class T2, class T3> int GetHeight(
    const ArrayRef5D<T1, T2, T3>& array);
template <class T1, class T2, class T3> int GetWidth(
    const ArrayRef2D<T1, T2, T3>& array);
template <class T1, class T2, class T3> int GetWidth(
    const ArrayRef3D<T1, T2, T3>& array);
template <class T1, class T2, class T3> int GetWidth(
    const ArrayRef4D<T1, T2, T3>& array);
template <class T1, class T2, class T3> int GetWidth(
    const ArrayRef5D<T1, T2, T3>& array);


#ifdef __SSE__
template <class T>
T* MallocAligned(unsigned int size);
template <class T>
void Memset(T* data, int byte_value, unsigned int size);

/** Allocator for 16-byte alligned memory. */
template <class T>
class SSEAllocator {
 public:
  static T* AllocateArray(int size);
  static void FreeArray(T* array);
};

/** Arrays with 16-byte aligned base pointer. */
typedef Array1D<float, SSEAllocator<float> > SSEArray1D;
typedef Array1D<v4f, SSEAllocator<v4f> > V4fArray1D;

typedef Array2D<float, SSEAllocator<float> > SSEArray2D;
typedef Array2D<v4f, SSEAllocator<v4f> > V4fArray2D;

typedef Array3D<float, SSEAllocator<float> > SSEArray3D;
typedef Array3D<v4f, SSEAllocator<v4f> > V4fArray3D;

typedef Array4D<float, SSEAllocator<float> > SSEArray4D;
typedef Array4D<v4f, SSEAllocator<v4f> > V4fArray4D;

typedef Array5D<float, SSEAllocator<float> > SSEArray5D;
typedef Array5D<v4f, SSEAllocator<v4f> > V4fArray5D;
#endif  // __SSE__


void CNormalizeArrayAcrossBand_UnitNorm(ArrayRef3D<float>* data);


/////////////////// TEMPLATE DEFINITIONS ///////////////////////////

template <class T>
T* StdAllocator<T>::AllocateArray(int size) {
  T* ptr = new T[size];
  return ptr;
}

template <class T>
void StdAllocator<T>::FreeArray(T* array) {
  delete[] array;
}

template <class T, class DATA, class REF>
ArrayRef1D<T, DATA, REF>::ArrayRef1D(int w0) : w0(w0) {
}

template <class T, class DATA, class REF>
ArrayRef1D<T, DATA, REF>::ArrayRef1D(DATA data, int w0) : data(data), w0(w0) {
}

template <class T, class DATA, class REF>
inline REF ArrayRef1D<T, DATA, REF>::operator[](int x) {
  DEBUG_ASSERT(x >= 0 && x < w0);
  return data[x];
}

template <class T, class DATA, class REF>
inline const REF ArrayRef1D<T, DATA, REF>::operator[](int x) const {
  DEBUG_ASSERT(x >= 0 && x < w0);
  return data[x];
}

template <class T, class DATA, class REF>
inline REF ArrayRef1D<T, DATA, REF>::operator()(int x) {
  return this->operator[](x);
}

template <class T, class DATA, class REF>
inline const REF ArrayRef1D<T, DATA, REF>::operator()(int x) const {
  return this->operator[](x);
}

template <class T, class DATA, class REF>
DATA ArrayRef1D<T, DATA, REF>::Get() {
  return data;
}

template <class T,class DATA,class REF>
const DATA ArrayRef1D<T, DATA, REF>::Get() const {
  return data;
}

template <class T, class DATA, class REF>
const int ArrayRef1D<T, DATA, REF>::Size() {
  return w0;
}

template <class T, class A>
Array1D<T, A>::Array1D(int w0) : ArrayRef1D<T>(w0) {
  this->data = A::AllocateArray(this->Size());
  ASSERT_NOT_NULL(this->data);
}

template <class T, class A>
Array1D<T, A>::~Array1D() {
  A::FreeArray(this->data);
}

template <class T, class DATA, class REF>
ArrayRef2D<T, DATA, REF>::ArrayRef2D(int w0, int w1) : w0(w0), w1(w1) {
}

template <class T, class DATA, class REF>
ArrayRef2D<T, DATA, REF>::ArrayRef2D(DATA data, int w0, int w1) :
    data(data), w0(w0), w1(w1) {
}

template <class T, class DATA, class REF>
inline REF ArrayRef2D<T, DATA, REF>::operator[](int x) {
  DEBUG_ASSERT(x >= 0 && x < Size());
  return data[x];
}

template <class T, class DATA, class REF>
inline REF ArrayRef2D<T, DATA, REF>::operator()(int x0, int x1) {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  int x = x1 + w1 * x0;
  return data[x];
}

template <class T, class DATA, class REF>
inline const REF ArrayRef2D<T, DATA, REF>::operator()(int x0, int x1) const {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  int x = x1 + w1 * x0;
  return data[x];
}

template <class T,class DATA,class REF>
DATA ArrayRef2D<T, DATA, REF>::Get() {
  return data;
}

template <class T,class DATA,class REF>
const DATA ArrayRef2D<T, DATA, REF>::Get() const {
  return data;
}

template <class T,class DATA,class REF>
const int ArrayRef2D<T, DATA, REF>::Size() {
  return w0 * w1;
}

template <class T, class A>
Array2D<T, A>::Array2D(int w0, int w1) : ArrayRef2D<T>(w0, w1) {
  this->data = A::AllocateArray(this->Size());
  ASSERT_NOT_NULL(this->data);
}

template <class T, class A>
Array2D<T, A>::~Array2D() {
  A::FreeArray(this->data);
}

template <class T, class DATA, class REF>
ArrayRef3D<T, DATA, REF>::ArrayRef3D(int w0, int w1, int w2) : w0(w0), w1(w1),
    w2(w2) {
}

template <class T, class DATA, class REF>
ArrayRef3D<T, DATA, REF>::ArrayRef3D(DATA data, int w0, int w1, int w2) :
        data(data), w0(w0), w1(w1), w2(w2) {
}

template <class T, class DATA, class REF>
inline REF ArrayRef3D<T, DATA, REF>::operator[](int x) {
  DEBUG_ASSERT(x >= 0 && x < Size());
  return data[x];
}

template <class T, class DATA, class REF>
inline REF ArrayRef3D<T, DATA, REF>::operator()(int x0, int x1, int x2) {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  DEBUG_ASSERT(x2 >= 0 && x2 < w2);
  int x = x2 + w2 * (x1 + w1 * x0);
  return data[x];
}

template <class T, class DATA, class REF>
inline const REF ArrayRef3D<T, DATA, REF>::operator()(int x0, int x1, int x2) const {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  DEBUG_ASSERT(x2 >= 0 && x2 < w2);
  int x = x2 + w2 * (x1 + w1 * x0);
  return data[x];
}

template <class T, class DATA, class REF>
DATA ArrayRef3D<T, DATA, REF>::Get() {
  return data;
}

template <class T,class DATA,class REF>
const DATA ArrayRef3D<T, DATA, REF>::Get() const {
  return data;
}

template <class T, class DATA, class REF>
const int ArrayRef3D<T, DATA, REF>::Size() {
  return w0 * w1 * w2;
}

template <class T, class A>
Array3D<T, A>::Array3D(int w0, int w1, int w2) : ArrayRef3D<T>(w0, w1, w2) {
  this->data = A::AllocateArray(this->Size());
  ASSERT_NOT_NULL(this->data);
}

template <class T, class A>
Array3D<T, A>::~Array3D() {
  A::FreeArray(this->data);
}


template <class T, class DATA, class REF>
ArrayRef4D<T, DATA, REF>::ArrayRef4D(int w0, int w1, int w2, int w3) :
    w0(w0), w1(w1), w2(w2), w3(w3) {
}

template <class T, class DATA, class REF>
ArrayRef4D<T, DATA, REF>::ArrayRef4D(DATA data, int w0, int w1, int w2,
    int w3) : data(data), w0(w0), w1(w1), w2(w2), w3(w3) {
}

template <class T, class DATA, class REF>
inline REF ArrayRef4D<T, DATA, REF>::operator[](int x) {
  DEBUG_ASSERT(x >= 0 && x < Size());
  return data[x];
}

template <class T, class DATA, class REF>
inline REF ArrayRef4D<T, DATA, REF>::operator()(int x0, int x1, int x2, int x3) {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  DEBUG_ASSERT(x2 >= 0 && x2 < w2);
  DEBUG_ASSERT(x3 >= 0 && x3 < w3);
  int x = x3 + w3 * (x2 + w2 * (x1 + w1 * x0));
  return data[x];
}

template <class T, class DATA, class REF>
inline const REF ArrayRef4D<T,DATA,REF>::operator()(int x0, int x1, int x2, int x3)
    const {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  DEBUG_ASSERT(x2 >= 0 && x2 < w2);
  DEBUG_ASSERT(x3 >= 0 && x3 < w3);
  int x = x3 + w3 * (x2 + w2 * (x1 + w1 * x0));
  return data[x];
}

template <class T, class DATA, class REF>
DATA ArrayRef4D<T,DATA,REF>::Get() {
  return data;
}

template <class T,class DATA,class REF>
const DATA ArrayRef4D<T, DATA, REF>::Get() const {
  return data;
}

template <class T, class DATA, class REF>
const int ArrayRef4D<T, DATA, REF>::Size() {
  return w0 * w1 * w2 * w3;
}

template <class T, class A>
Array4D<T, A>::Array4D(int w0, int w1, int w2, int w3) :
    ArrayRef4D<T>(w0, w1, w2, w3) {
  this->data = A::AllocateArray(this->Size());
  ASSERT_NOT_NULL(this->data);
}

template <class T, class A>
Array4D<T,A>::~Array4D() {
  A::FreeArray(this->data);
}

template <class T, class DATA, class REF>
ArrayRef5D<T, DATA, REF>::ArrayRef5D(int w0, int w1, int w2, int w3, int w4) :
    w0(w0), w1(w1), w2(w2), w3(w3), w4(w4) {
}

template <class T, class DATA, class REF>
ArrayRef5D<T, DATA, REF>::ArrayRef5D(DATA data, int w0, int w1, int w2, int w3,
    int w4) : data(data), w0(w0), w1(w1), w2(w2), w3(w3), w4(w4) {
}

template <class T, class DATA, class REF>
inline REF ArrayRef5D<T, DATA, REF>::operator[](int x) {
  DEBUG_ASSERT(x >= 0 && x < Size());
  return data[x];
}

template <class T, class DATA, class REF>
inline REF ArrayRef5D<T, DATA, REF>::operator()(int x0, int x1, int x2, int x3,
    int x4) {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  DEBUG_ASSERT(x2 >= 0 && x2 < w2);
  DEBUG_ASSERT(x3 >= 0 && x3 < w3);
  DEBUG_ASSERT(x4 >= 0 && x4 < w4);
  int x = x4 + w4 * (x3 + w3 * (x2 + w2 * (x1 + w1 * x0)));
  return data[x];
}

template <class T,class DATA,class REF>
inline const REF ArrayRef5D<T, DATA, REF>::operator()(int x0, int x1, int x2, int x3,
    int x4) const {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  DEBUG_ASSERT(x2 >= 0 && x2 < w2);
  DEBUG_ASSERT(x3 >= 0 && x3 < w3);
  DEBUG_ASSERT(x4 >= 0 && x4 < w4);
  int x = x4 + w4 * (x3 + w3 * (x2 + w2 * (x1 + w1 * x0)));
  return data[x];
}

template <class T, class DATA, class REF>
DATA ArrayRef5D<T, DATA, REF>::Get() {
  return data;
}

template <class T,class DATA,class REF>
const DATA ArrayRef5D<T, DATA, REF>::Get() const {
  return data;
}

template <class T, class DATA, class REF>
const int ArrayRef5D<T, DATA, REF>::Size() {
  return w0 * w1 * w2 * w3 * w4;
}

template <class T, class A>
Array5D<T, A>::Array5D(int w0, int w1, int w2, int w3, int w4) : ArrayRef5D<T>(
    w0, w1, w2, w3, w4) {
  this->data = A::AllocateArray(this->Size());
  ASSERT_NOT_NULL(this->data);
}

template <class T, class A>
Array5D<T, A>::~Array5D() {
  A::FreeArray(this->data);
}

inline int Index2D::operator()(int x0, int x1) {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  return x1 + w1 * x0;
}

inline int Index3D::operator()(int x0, int x1, int x2) {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  DEBUG_ASSERT(x2 >= 0 && x2 < w2);
  return x2 + w2 * (x1 + w1 * x0);
}

inline int Index4D::operator()(int x0, int x1, int x2, int x3) {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  DEBUG_ASSERT(x2 >= 0 && x2 < w2);
  DEBUG_ASSERT(x3 >= 0 && x3 < w3);
  return x3 + w3 * (x2 + w2 * (x1 + w1 * x0));
}

inline int Index5D::operator()(int x0, int x1, int x2, int x3, int x4) {
  DEBUG_ASSERT(x0 >= 0 && x0 < w0);
  DEBUG_ASSERT(x1 >= 0 && x1 < w1);
  DEBUG_ASSERT(x2 >= 0 && x2 < w2);
  DEBUG_ASSERT(x3 >= 0 && x3 < w3);
  DEBUG_ASSERT(x4 >= 0 && x4 < w4);
  return x4 + w4 * (x3 + w3 * (x2 + w2 * (x1 + w1 * x0)));
}

template <class ARRAY>
void Memset(ARRAY* array, int byte_value) {
  // NOTE: Callee scales by size of entry.
  Memset(array->Get(), byte_value, array->Size());
}

template <class T1, class T2, class T3>
int GetHeight(const ArrayRef2D<T1, T2, T3>& array) {
  return array.w0;
}

template <class T1, class T2, class T3>
int GetHeight(const ArrayRef3D<T1, T2, T3>& array) {
  return array.w1;
}

template <class T1, class T2, class T3>
int GetHeight(const ArrayRef4D<T1, T2, T3>& array) {
  return array.w2;
}

template <class T1, class T2, class T3>
int GetHeight(const ArrayRef5D<T1, T2, T3>& array) {
  return array.w3;
}

template <class T1, class T2, class T3>
int GetWidth(const ArrayRef2D<T1, T2, T3>& array) {
  return array.w1;
}

template <class T1, class T2, class T3>
int GetWidth(const ArrayRef3D<T1, T2, T3>& array) {
  return array.w2;
}

template <class T1, class T2, class T3>
int GetWidth(const ArrayRef4D<T1, T2, T3>& array) {
  return array.w3;
}

template <class T1, class T2, class T3>
int GetWidth(const ArrayRef5D<T1, T2, T3>& array) {
  return array.w4;
}


#ifdef __SSE__
template <class T>
T* MallocAligned(unsigned int size) {
  T* data = static_cast<T*>( _mm_malloc(size * sizeof(T), 16) );
  ASSERT_NOT_NULL(data);
  return data;
}

template <class T>
void Memset(T* data, int byte_value, unsigned int size) {
  memset(data, byte_value, size * sizeof(T));
}

template <class T>
T* SSEAllocator<T>::AllocateArray(int size) {
  T* ptr = reinterpret_cast<T*>( _mm_malloc(sizeof(T) * size, 16) );
  return ptr;
}

template <class T>
void SSEAllocator<T>::FreeArray(T* array) {
  _mm_free(array);
}
#endif  // __SSE__

#endif  // __ARRAY_H__
