# This module implements the "Viz2" model used for the GCNC 2011 experiments.

from glimpse import core, util
from glimpse.core import misc
import numpy as np
import Image
import random

# Create a 2-part, HMAX-like hierarchy of S+C layers.

R_KWIDTH = 15
S1_KWIDTH = 11
C1_KWIDTH = 5
S2_KWIDTH = 7
NUM_SCALES = 4
NUM_ORIENTATIONS = 8

def LoadAndPreprocess(img_fname):
  image = core.ImageToInputArray(Image.open(img_fname))
  retina = core.BuildRetinalLayer(image, kwidth = R_KWIDTH)
  return retina.reshape((1,) + retina.shape)

def MakeS1Kernels():
  # Kernel array is 5-dimensional: scale, orientation, phase, y, x
  all_s1_ks = core.MakeMultiScaleGaborKernels(kwidth = S1_KWIDTH,
      num_scales = NUM_SCALES, num_orientations = NUM_ORIENTATIONS,
      num_phases = 2, shift_orientations = True, scale_norm = True)
  # Reshape kernel array to be 4-D: scale, index, 1, y, x
  return all_s1_ks.reshape((NUM_SCALES, -1, 1, S1_KWIDTH, S1_KWIDTH))

def NormRbf(data, kernels):
  assert len(data.shape) == 3
  assert len(kernels.shape) == 4
  return misc.BuildSimpleLayer(data, kernels = kernels, scaling = 2)

def PoolC1(s1):
  # Take maximum value over phase
  s1 = s1.reshape((NUM_ORIENTATIONS, -1) + s1.shape[-2:])
  s1 = s1.max(1)
  c1, coords = misc.BuildComplexLayer(s1, kwidth = C1_KWIDTH,
      kheight = C1_KWIDTH, scaling = 2)
  return c1

def PoolC2(s2):
  # C2 is global maximum over S2 prototypes
  s2 = s2.reshape(s2.shape[0], -1)
  return s2.max(1)

def BuildThroughC1(img_fname):
  all_s1_ks = MakeS1Kernels()
  retina = LoadAndPreprocess(img_fname)
  for s1_ks in all_s1_ks:
    s1 = NormRbf(retina, s1_ks)
    c1 = PoolC1(s1)
    yield c1
  raise StopIteration

def Build(img_fname, s2_ks_fname):
  s2_ks = util.Load(s2_ks_fname)
  assert len(s2_ks.shape) == 4 and s2_ks.shape[1] == NUM_ORIENTATIONS, \
      "S2 prototypes have wrong shape: %s" % (s2_ks.shape,)
  c2s = []
  for c1 in BuildThroughC1(img_fname):
    s2 = NormRbf(c1, s2_ks)
    c2 = PoolC2(s2)
    c2s.append(c2)
  # IT is max over scales
  it = np.array(c2s).max(0)
  return it

def ImprintPrototypes(img_fname, num_prototypes):
  c1s = list(BuildThroughC1(img_fname))
  prototypes = np.empty([num_prototypes, NUM_ORIENTATIONS, S2_KWIDTH,
      S2_KWIDTH], core.activation_dtype)
  for proto in prototypes:
    scale = random.randint(0, NUM_SCALES - 1)
    c1 = c1s[scale]
    height, width = c1.shape[-2:]
    y = random.randint(0, height - S2_KWIDTH)
    x = random.randint(0, width - S2_KWIDTH)
    proto[:] = c1[ :, y : y + S2_KWIDTH, x : x + S2_KWIDTH ]
    util.ScaleUnitNorm(proto)
  assert np.allclose(np.array(map(np.linalg.norm, prototypes)), 1), \
      "Internal error: S2 prototypes are not normalized"
  assert not np.isnan(prototypes).any(), \
      "Internal error: found NaN in imprinted prototype."
  return prototypes

if __name__ == "__main__":
  import sys
  if len(sys.argv) < 4:
    sys.exit("usage: OP IMAGE PROTOS [OPTIONS] > IT-ACTIVITY\n"
        "where OP is one of IMPRINT or TRANSFORM")
  op, img_fname, protos_fname = sys.argv[1:4]
  if op.upper() == "TRANSFORM":
    util.Store(Build(img_fname, protos_fname), sys.stdout)
  else:
    if len(sys.argv) < 5:
      sys.exit("IMPRINT requires number of prototypes as an extra option")
    num_prototypes = int(sys.argv[4])
    util.Store(ImprintPrototypes(img_fname, num_prototypes), sys.stdout)
