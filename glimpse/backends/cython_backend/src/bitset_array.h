/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#ifndef __BITSET_ARRAY_H__
#define __BITSET_ARRAY_H__

#include "array.h"

typedef unsigned char byte;

/*
might make more sense to have BitsetArray be an array of bits?
but we don't currently have an array of bits, just a set of bits.
so the name should be Bitset, and BitsetArray. maybe the capitalization will
make the structure more clear?
*/

// A BitsetRef is a pointer to a single existing bitset. Use an IndexND to treat
// this bitset as an ND array of bits.
class BitsetRef {
 private:
  byte* const data;
  const int num_bits;
 public:
  BitsetRef(byte* data, int num_bits);
  void Set(int idx);
  void Unset(int idx);
  bool IsSet(int idx) const;
  void Clear();
  void Fill();
  /** Get number of set bits. */
  int PopCount() const;
  int Size() const;
};

// std::ostream& operator<<(std::ostream& o, const BitsetRef& bitset);

// A BitsetArrayRef is a pointer to an existing one-dimensional list of Bitset
// data. The Bitset data is packed together in a long byte array, but elements
// of the BitsetArrayRef list provide a BitsetRef "view" of one part of the
// packed data. To get more complex array behavior (i.e., shapes of more than
// one dimension), create ArrayRefND objects that are backed by a
// BitsetArrayRef. For example, see the BitsetArrayRefND objects below.
class BitsetArrayRef {
 protected:
  byte* data;
  BitsetArrayRef(int num_sets, int num_bits_per_set);
 public:
  const int num_bits_per_set;
  const int num_bytes_per_set;
  const int num_sets;
  BitsetArrayRef(byte* data, int num_sets, int num_bits_per_set);
  BitsetRef operator[](int bitset_index);
  const BitsetRef operator[](int bitset_index) const;
  byte* GetBase();
  const byte* GetBase() const;
  int GetNumBitsPerSet() const;
  int GetNumBytesPerSet() const;
  /** Return size of array in bitsets. */
  int Size() const;
};

// ArrayRefND wrappers pointing to BitsetArrayRef objects. For example, a
// BitsetArrayRef3D is a 3D array (provided by ArrayRef3D), where each element
// is 1D bitset (provided by BitsetRef).
typedef ArrayRef1D<BitsetRef, BitsetArrayRef, BitsetRef> BitsetArrayRef1D;
typedef ArrayRef2D<BitsetRef, BitsetArrayRef, BitsetRef> BitsetArrayRef2D;
typedef ArrayRef3D<BitsetRef, BitsetArrayRef, BitsetRef> BitsetArrayRef3D;
typedef ArrayRef4D<BitsetRef, BitsetArrayRef, BitsetRef> BitsetArrayRef4D;
typedef ArrayRef5D<BitsetRef, BitsetArrayRef, BitsetRef> BitsetArrayRef5D;

// A BitsetArray is a list of bitsets (i.e., not a pointer, this object owns the
// memory for all bitsets.
class BitsetArray : public BitsetArrayRef {
 public:
  BitsetArray(int num_sets, int num_bits_per_set);
  ~BitsetArray();
};

/// Inline methods indended to be used only by other bitset array methods. ///

namespace BitsetArray_detail {
inline int BitsToBytes(int num_bits) {
  int num_bytes = num_bits / 8;
  if (num_bits % 8) {
    num_bytes++;
  }
  return num_bytes;
}
inline int GetByte(int idx) {
  return idx / 8;
}
inline int GetBit(int idx) {
  return 8 - (idx % 8 + 1);
}
}

/// Inline definitions of BitsetRef and BitsetArrayRef methods. ///

inline void BitsetRef::Set(int idx) {
  DEBUG_ASSERT(idx >= 0 && idx < num_bits);
  int byte_idx = BitsetArray_detail::GetByte(idx);
  int bit_idx = BitsetArray_detail::GetBit(idx);
  byte b = 1;
  b <<= bit_idx;
  data[byte_idx] |= b;
}

inline void BitsetRef::Unset(int idx) {
  DEBUG_ASSERT(idx >= 0 && idx < num_bits);
  int byte_idx = BitsetArray_detail::GetByte(idx);
  int bit_idx = BitsetArray_detail::GetBit(idx);
  byte b = 1;
  b <<= bit_idx;
  data[byte_idx] &= ~b;
}

inline bool BitsetRef::IsSet(int idx) const {
  DEBUG_ASSERT(idx >= 0 && idx < num_bits);
  int byte_idx = BitsetArray_detail::GetByte(idx);
  int bit_idx = BitsetArray_detail::GetBit(idx);
  byte b = 1;
  b <<= bit_idx;
  return data[byte_idx] & b;
}

inline void BitsetRef::Clear() {
  byte b = 0;
  memset(static_cast<void*>(data), b, BitsetArray_detail::BitsToBytes(num_bits));
}

inline void BitsetRef::Fill() {
  byte b = 0;
  memset(static_cast<void*>(data), ~b, BitsetArray_detail::BitsToBytes(num_bits));
}

/** XXX This relies on pad bits being zero. */
inline int BitsetRef::PopCount() const {
  int num_whole_bytes = num_bits / 8;
  int cnt = 0;
  // Compute number of set bits in all but last byte
  for (int i=0; i < num_whole_bytes; i++) {
    cnt += __builtin_popcount(data[i]);
  }
  // Count number of set bits in last byte, if needed
  if (num_bits % 8) {
    byte mask = 1;
    mask <<= BitsetArray_detail::GetBit(num_bits-1);
    mask -= 1;
    mask = ~mask;
    byte last_byte = data[num_whole_bytes];
    cnt += __builtin_popcount(last_byte & mask);
  }
  return cnt;
}

inline int BitsetRef::Size() const {
  return num_bits;
}

inline BitsetRef BitsetArrayRef::operator[](int bitset_index) {
  DEBUG_ASSERT(bitset_index >= 0 && bitset_index < num_sets);
  byte* base = data + bitset_index * num_bytes_per_set;
  return BitsetRef(base, num_bits_per_set);
}

inline const BitsetRef BitsetArrayRef::operator[](int bitset_index) const {
  DEBUG_ASSERT(bitset_index >= 0 && bitset_index < num_sets);
  byte* base = data + bitset_index * num_bytes_per_set;
  return BitsetRef(base, num_bits_per_set);
}

inline byte* BitsetArrayRef::GetBase() {
  return data;
}

inline const byte* BitsetArrayRef::GetBase() const {
  return data;
}

inline int BitsetArrayRef::GetNumBitsPerSet() const {
  return num_bits_per_set;
}

inline int BitsetArrayRef::GetNumBytesPerSet() const {
  return num_bytes_per_set;
}

inline int BitsetArrayRef::Size() const {
  return num_sets;
}

#endif // __BITSET_ARRAY_H__
