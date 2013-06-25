/*******************************************************************************
 * Copyright (c) 2011 Mick Thomure                                             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Please see the file COPYING in this distribution for usage terms.           *
 ******************************************************************************/

#include "bitset_array.h"
#include <cstring>
#include "util.h"
// #include <iostream>


///////////////////// BitsetRef ///////////////////////////

BitsetRef::BitsetRef(byte* data, int num_bits) :
  data(data), num_bits(num_bits) {
}

// std::ostream& operator<<(std::ostream& o, const BitsetRef& bitset) {
  // for (int i=0; i < bitset.Size(); i++) {
    // if (i > 0 && i % 8 == 0) {
      // o << " ";
    // }
    // if (bitset.IsSet(i)) {
      // o << "1";
    // } else {
      // o << "0";
    // }
  // }
  // return o;
// }

///////////////////// BitsetArayRef ///////////////////////////

BitsetArrayRef::BitsetArrayRef(int num_sets, int num_bits_per_set) :
  data(NULL),
  num_bits_per_set(num_bits_per_set),
  num_bytes_per_set(BitsetArray_detail::BitsToBytes(num_bits_per_set)),
  num_sets(num_sets) {
}

BitsetArrayRef::BitsetArrayRef(byte* data, int num_sets, int num_bits_per_set) :
    data(data),
    num_bits_per_set(num_bits_per_set),
    num_bytes_per_set(BitsetArray_detail::BitsToBytes(num_bits_per_set)),
    num_sets(num_sets) {
  if (num_sets > 0) {
    ASSERT_NOT_NULL(data);
  }
}

///////////////////// BitsetAray ///////////////////////////

BitsetArray::BitsetArray(int num_sets, int num_bits_per_set) :
    BitsetArrayRef(num_sets, num_bits_per_set) {
  int num_total_bytes = num_sets * num_bytes_per_set;
  data = new byte[num_total_bytes];
  ASSERT_NOT_NULL(data);
  memset(data, 0, num_total_bytes);
}

BitsetArray::~BitsetArray() {
  delete[] data;
}
