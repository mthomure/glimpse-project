# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

import numpy as np

from glimpse.util import kernel
from glimpse.util import traits
from glimpse.util import ACTIVATION_DTYPE

class S1Params():
  """Configuration for a single S1 scale."""

  gamma = 0.3
  thetas = (0, 45, 90, 135)
  num_orientations = len(thetas)

  def __init__(self, kwidth, sigma, lambda_):
    self.kwidth, self.sigma, self.lambda_ = kwidth, sigma, lambda_
    self._kernels = None

  @property
  def kernels(self):
    """Get the kernel array for this configuration.

    :rtype: 4D ndarray of float

    """
    if self._kernels == None:
      kernels = [ kernel.MakeGaborKernel(self.kwidth, theta,
          gamma = S1Params.gamma, sigma = self.sigma, phi = 0,
          lambda_ = self.lambda_, scale_norm = True)
          for theta in S1Params.thetas ]
      self._kernels = np.array(kernels, dtype = ACTIVATION_DTYPE)
    return self._kernels

class V1Params():
  """Parameters for a single V1 scale band.

  This gives the configuration for a single C1 scale, and a subset of S1 scales.

  """

  def __init__(self, c1_kwidth, c1_overlap, *s1_params):
    self.c1_kwidth, self.c1_overlap, self.s1_params = c1_kwidth, c1_overlap, \
        s1_params

  @property
  def c1_scaling(self):
    # The step size is the fraction of one neighborhood that does not overlap
    # the next.
    return self.c1_kwidth - self.c1_overlap

class Params(traits.HasTraits):
  """Parameter container for the :class:`hmax model
  <glimpse.models.hmax.model.Model>`.

  """

  def __init__(self):
    #: Fixed parameters for S1 filters, arranged in bands of 2 scales each. See
    #: Table 1 of Serre et al (2007).
    self.v1 = [
      V1Params(8,   4, S1Params( 7,  2.8,  3.5), S1Params( 9,  3.6,  4.6)),
      V1Params(10,  5, S1Params(11,  4.5,  5.6), S1Params(13,  5.4,  6.8)),
      V1Params(12,  6, S1Params(15,  6.3,  7.9), S1Params(17,  7.3,  9.1)),
      V1Params(14,  7, S1Params(19,  8.2, 10.3), S1Params(21,  9.2, 11.5)),
      V1Params(16,  8, S1Params(23, 10.2, 12.7), S1Params(25, 11.3, 14.1)),
      V1Params(18,  9, S1Params(27, 12.3, 15.4), S1Params(29, 13.4, 16.8)),
      V1Params(20, 10, S1Params(31, 14.6, 18.2), S1Params(33, 15.8, 19.7)),
      V1Params(22, 11, S1Params(35, 17.0, 21.2), S1Params(37, 18.2, 22.8)),
    ]
    #: Fixed value for S2 beta parameter.
    self.s2_beta = 1.0
    #: Fixed width for the S2 kernels.
    self.s2_kwidths = (4, 8, 12, 16)

  def __str__(self):
    # Get list of all traits.
    traits = self.traits().keys()
    # Remove special entries from the HasTraits object.
    traits = filter((lambda t: not t.startswith("trait_")), traits)
    # Format set of traits as a string.
    return "Params(\n  %s\n)" % "\n  ".join("%s = %s" % (tn,
        getattr(self, tn)) for tn in traits)

  __repr__ = __str__
