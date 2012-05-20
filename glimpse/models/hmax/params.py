# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.util import traits

class Params(traits.HasStrictTraits):
  """Parameter container for the :class:`hmax model
  <glimpse.models.hmax.model.Model>`.

  """

  def __str__(self):
    # Get list of all traits.
    traits = self.traits().keys()
    # Remove special entries from the HasTraits object.
    traits = filter((lambda t: not t.startswith("trait_")), traits)
    # Format set of traits as a string.
    return "Params(\n  %s\n)" % "\n  ".join("%s = %s" % (tn,
        getattr(self, tn)) for tn in traits)

  __repr__ = __str__
