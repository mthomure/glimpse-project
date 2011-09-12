
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Classes and functions for computing random sets of numbers.
#

import numpy

class HistogramSampler(object):
  """Draw samples according to a distribution inferred from the histogram of a
  given dataset."""

  def __init__(self, data, nbins = 100):
    magnitudes, edges = numpy.histogram(data.flat, nbins)
    magnitudes = magnitudes.astype(numpy.float) / magnitudes.sum()
    # Transform a vector of relative magnitudes (one entry per bin) to a vector
    # of repeated index groups. Group I has N entries, where N is proportional
    # to the magnitude of bin I.
    group_sizes = (magnitudes * 4 * nbins).astype(numpy.int)
    # Ignore groups with zero elements.
    idx = group_sizes.nonzero()[0]
    cnt = group_sizes[ idx ]  # Ideally this would be 100, but usually it's
                              # less.
    # Create groups of replicated indices.
    groups = numpy.empty(cnt.sum(), numpy.int)
    last_idx = 0
    for i, c in zip(idx, cnt):
      next_idx = last_idx + c
      groups[last_idx : next_idx] = i
      last_idx = next_idx
    self.edges = edges
    self.groups = groups
    self.bin_width = edges[1] - edges[0]

  def Sample(self, nsamples = 1):
    idx = self.groups[ numpy.random.randint(0, len(self.groups), nsamples) ]
    offsets = numpy.random.random_sample(nsamples) * self.bin_width
    offsets += self.edges[ idx ]
    return offsets
