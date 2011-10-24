
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Classes and functions for computing random sets of numbers.
#

import numpy as np

class HistogramSampler(object):
  """Given repeated observations of a single random variable, first model the
  probability distribution governing the variable using a histogram, and then
  generate new variates according to this distribution.

  This sampler trades space for time by approximating the cumulative
  histogram as a single linear array in memory, where the value of a histogram
  bin is represented repeatedly according to its magnitude. New variates are
  generated by sampling uniformly from the indices of this array, and returning
  the edge value of the corresponding bin. The accuracy of the sampler is
  governed by both the number of bins in the histogram, and the number of
  elements in the cumulative distribution (cum-dist) array.
  """

  # Relative drop in range of values between observed and generated variates.
  range_error = 0

  def __init__(self, data, nbins = 100, resolution = 0.0025):
    """
    Construct a new sampler object.
    data -- (1-D) array of observations for a single random variable
    nbins -- (positive int) number of bins to use when generating the histogram
    resolution -- (float in [0,1)) resolution of each element of the cum-dist
                  array. For example, a resolution of 0.25 means that a
                  histogram bin is represented in the cum-dist array only when
                  its magnitude is at least 25% of the total mass (i.e., that at
                  least 1/4 of the total observations fell in the bin's
                  interval). In this case, the cum-dist array requires only four
                  elements. On the other hand, a resolution of 0.0025 means that
                  a histogram bin needs to contain only 0.25% of the total mass
                  to be represented in the cum-dist array, which now requires
                  400 elements.
    """
    # Bin the data.
    magnitudes, edges = np.histogram(data, nbins)
    bin_width = edges[1] - edges[0]
    # Number of cum-dist elements per bin. This is the normalized bin
    # magnitudes, multiplied by the inverse resolution (which gives the total
    # number of elements in the cumulative distribution array).
    bin_sizes = magnitudes.astype(np.float) / (magnitudes.sum() * resolution)
    # Discard any bin that accounts for less than one full element in the
    # cum-dist array.
    valid_indices = np.where(bin_sizes >= 1)[0]
    if valid_indices.size == 0:
      # XXX Is there a heuristic for choosing the resolution from, for example,
      # the number of bins in the histogram?
      raise ValueError("Resolution is too low. Cumulative distribution array is"
          " empty.")
    bin_sizes = bin_sizes[valid_indices].astype(int)
    # Fill cumulative distribution array.
    cum_dist = np.empty(bin_sizes.sum(), np.int)
    last = 0
    for bin_idx, size in zip(valid_indices, bin_sizes):
      next = last + size
      # Create (size) copies of the current bin index.
      cum_dist[last : next] = bin_idx
      last = next
    self.edges = edges
    self.cum_dist = cum_dist
    self.bin_width = bin_width
    # Calculate approximation error.
    target_range = edges[0], edges[-1] + bin_width
    target = target_range[1] - target_range[0]
    actual_range = edges[valid_indices[0]], edges[valid_indices[-1]] + bin_width
    actual = actual_range[1] - actual_range[0]
    self.range_error = (target - actual) / target

  def Sample(self, nsamples = 1):
    """Generate variates according to the modelled distribution.
    nsamples -- (positive int) the number of variates to generate
    """
    # Sample uniformly from the cumulative distribution array.
    cum_dist_indices = np.random.randint(0, len(self.cum_dist), nsamples)
    bin_indices = self.cum_dist[cum_dist_indices]
    # Lookup the corresponding edge values for each bin.
    offsets = self.edges[ bin_indices ]
    # Add a small delta, so that returned samples are drawn uniformly from the
    # corresponding bins.
    offsets += np.random.random_sample(nsamples) * self.bin_width
    return offsets
