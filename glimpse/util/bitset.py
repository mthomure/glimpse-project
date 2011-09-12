
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Classes and functions for arrays of bitsets (i.e., sets of binary values).
#

import math
import numpy
import operator

class BitsetArray(object):
  """A helper class for wrapping a byte array containing an ND-array of
  bitsets."""
  def __init__(self, memory, bitset_shape):
    self.memory = memory
    self.bitset_shape = bitset_shape
    assert reduce(operator.mul, bitset_shape) <= memory.shape[-1] * 8, \
        "Memory size (%d bytes) too small for bitset shape (%s)" % \
        (memory.shape[-1], bitset_shape,)
  def Reshape(self, array_shape):
    """Create a new view of this object's memory, in which the array of bitsets
    has a different shape. The length of the component bitsets is unchanged."""
    memory = self.memory.reshape(array_shape + (self.memory.shape[-1]))
    return BitsetArray(memory, self.bitset_shape)
  def ReshapeBitset(self, bitset_shape):
    """Create a new view of this object's memory, in which the component bitsets
    have a different length. The shape of the array of bitsets is unchanged."""
    return BitsetArray(self.memory, bitset_shape)
  @property
  def shape(self):
    """Get the shape of the array."""
    return self.memory.shape[:-1]
  @property
  def size(self):
    """Get the total number of bits in the array of bitsets. If the shape of the
    array is (2, 3) and each bitset has 4 elements, then the size is 24 bits."""
    return reduce(operator.mul, self.memory.shape[:-1]) * \
        reduce(operator.mul, self.bitset_shape)
  def ToArray(self, array_idx = None):
    """Create a boolean numpy array from this packed bitset array."""
    nbits_per_set = reduce(operator.mul, self.bitset_shape)
    if array_idx == None:
      memory = self.memory
      shape = self.memory.shape[:-1] + tuple(self.bitset_shape)
    else:
      memory = self.memory[array_idx]
      shape = self.bitset_shape
    # Convert uchar-packed representation to a seperate byte per bit.
    bits = numpy.unpackbits(memory, -1)
    # Byte array may have padded junk on the end. Remove it.
    bits.shape = (-1, bits.shape[-1])
    bits = bits[:, 0 : nbits_per_set]
    # Reshape byte array.
    bits.shape = shape
    return bits
  def FromArray(self, bits, array_idx = None):
    """Set values of this packed bitset array from a boolean numpy array."""
    if array_idx == None:
      size = self.size
      memory = self.memory.reshape(-1)
      array_shape = self.memory.shape[:-1]
      bitset_shape = array_shape + (-1,)
    else:
      nbits_per_set = reduce(operator.mul, self.bitset_shape)
      size = nbits
      memory = self.memory[array_idx]
      bitset_shape = (-1,)

    assert(bits.size == self.size)
    bits = bits.reshape(bitset_shape)
    bits = numpy.packbits(bits, -1)
    memory[:] = bits.flat
    # memory = self.memory.reshape(-1)
    # bits = bits.reshape(self.array_shape + (-1,))
    # bits = numpy.packbits(bits, -1)
    # bits.shape = -1
    # memory[:] = bits[:]
  # def SetBit(self, array_idx, bitset_idx, value = True):
    # self.memory[array_idx]

def MakeBitsetArrayMemory(array_shape, bitset_shape):
  """Create a byte array whose size can support an array of bitsets with the
  given shape.
    array_shape: dimensions of array whose elements are bitsets
    bitset_shape: dimensions of bit array stored in each bitset"""
  num_bits = reduce(operator.mul, bitset_shape)
  num_bytes = math.ceil(num_bits / 8.0)
  return numpy.zeros(list(array_shape) + [num_bytes], numpy.uint8)

def MakeBitsetArray(array_shape, bitset_shape, memory = None):
  """Create an ND-array of bitsets with the given shape."""
  if memory == None:
    memory = MakeBitsetArrayMemory(array_shape, bitset_shape)
  return BitsetArray(memory, bitset_shape)

def FromArray(array, bitset_dims):
  """Create a bitset from a boolean array.
  bitset_dims - number of final dimensions of array that correspond to a single
                bitset"""
  array = array.astype(numpy.uint8)
  array_shape = array.shape[:-bitset_dims]
  bitset_shape = array.shape[-bitset_dims:]
  ba = MakeBitsetArray(array_shape, bitset_shape)
  ba.FromArray(array)
  return ba

