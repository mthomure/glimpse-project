"""

Load a binary matrix as used by Huang & LeCun to encode the NORB image dataset.

The files are stored in the so-called 'binary matrix' file format, which is a
simple format for vectors and multidimensional matrices of various element
types. Binary matrix files begin with a file header which describes the type
and size of the matrix, and then comes the binary image of the matrix.

The header is best described by a C structure:

struct header {
int magic; // 4 bytes
int ndim; // 4 bytes, little endian
int dim[3];
};

Note that when the matrix has less than 3 dimensions, say, it's a 1D vector,
then dim[1] and dim[2] are both 1. When the matrix has more than 3 dimensions,
the header will be followed by further dimension size information. Otherwise,
after the file header comes the matrix data, which is stored with the index in
the last dimension changes the fastest.

The magic number encodes the element type of the matrix:
- 0x1E3D4C51 for a single precision matrix
- 0x1E3D4C52 for a packed matrix
- 0x1E3D4C53 for a double precision matrix
- 0x1E3D4C54 for an integer matrix
- 0x1E3D4C55 for a byte matrix
- 0x1E3D4C56 for a short matrix

Since the files are generated on an Intel machine, they use the little-endian
scheme to encode the 4-byte integers. Pay attention when you read the files on
machines that use big-endian.

The '-dat' files store a 4D tensor of dimensions 24300x2x96x96. Each files has
24,300 image pairs, (obviously, each pair has 2 images), and each image is
96x96 pixels. The '-cat' files store a 2D vector of dimension 24,300x1. The
'-info' files store a 2D matrix of dimensions 24300x4.

"""

import struct
import numpy as np

MAGIC_SINGLE = 0x1E3D4C51 # for a single precision matrix
MAGIC_PACKED = 0x1E3D4C52 # for a packed matrix
MAGIC_DOUBLE = 0x1E3D4C53 # for a double precision matrix
MAGIC_INT = 0x1E3D4C54 # for an integer matrix
MAGIC_BYTE = 0x1E3D4C55 # for a byte matrix
MAGIC_SHORT = 0x1E3D4C56 # for a short matrix

def LoadBinaryMatrix(fh):
  """Create a mmap view of a binary matrix.

  :param fh: Path to file, or open file handle.
  :type fh: string or file-like object
  :rtype: numpy.memmap object

  """
  if isinstance(fh, basestring):
    fh = open(fh, 'rb')
  else:
    fh.seek(0)
  magic, ndim = struct.unpack('ii', fh.read(8))
  offset = max(20, 4 * (2 + ndim))  # offset is at least 20 bytes
  dim = struct.unpack('iii', fh.read(12))
  if ndim > 3:
    dim += struct.unpack('i' * (ndim-3), fh.read(4 * (ndim-3)))
  if magic == MAGIC_PACKED:
    raise ValueError("Unable to parse matrices with element type: packed")
  dtype = { MAGIC_SINGLE : np.single, MAGIC_DOUBLE : np.double,
            MAGIC_INT : np.int32, MAGIC_BYTE : np.uint8,
            MAGIC_SHORT : np.short }[magic]
  return np.memmap(fh, dtype, mode = 'r', offset = offset).reshape(dim)
