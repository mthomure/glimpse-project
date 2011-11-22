# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse import util

ALL_OPTIONS = [
  ('retina_bias', "Term added to standard deviation of local window"),
  ('retina_enabled', "Indicates whether the retinal layer is used"),
  ('retina_kwidth', "Spatial width of input neighborhood for retinal units"),

  ('s1_beta', "Beta parameter of RBF for S1 cells"),
  ('s1_bias', "Term added to the norm of the input vector"),
  ('s1_kwidth', "Spatial width of input neighborhood for S1 units"),
  ('s1_num_orientations', "Number of different S1 Gabor orientations"),
  ('s1_num_phases', """Number of different phases for S1 Gabors. Using two
                    phases corresponds to find a light bar on a dark
                    background and vice versa"""),
  ('s1_scaling', """Subsampling factor (e.g., setting s1_scaling=2 will result
                 in an output array that is 1/2 the width and half the
                 height of the input array)"""),
  ('s1_shift_orientations', "Rotate Gabors by a small positive angle"),

  ('c1_kwidth', "Spatial width of input neighborhood for C1 units"),
  ('c1_scaling', "Subsampling factor"),
  ('c1_whiten', "Whether to normalize the total energy at each C1 location"),

  ('s2_beta', "Beta parameter of RBF for S1 cells"),
  ('s2_bias', "Additive term combined with input window norm"),
  ('s2_kwidth', "Spatial width of input neighborhood for S2 units"),
  ('s2_scaling', "Subsampling factor"),

  ('c2_kwidth', "Spatial width of input neighborhood for C2 units"),
  ('c2_scaling', "Subsampling factor"),

  ('num_scales', "Number of different scale bands"),
]

class Viz2Params(object):

  def __init__(self, **args):
    self._params = MakeDefaultParamDict()
    for name, value in args.items():
      self[name] = value

  def __repr__(self):
    return str(self._params)

  def __eq__(self, params):
    return params != None and self._params == params._params

  def __getitem__(self, name):
    if name not in self._params:
      raise ValueError("Unknown Viz2 param: %s" % name)
    return self._params[name]

  def __setitem__(self, name, value):
    assert name in self._params
    self._params[name] = value

  def __getstate__(self):
    return self._params

  def __setstate__(self, state):
    self._params = MakeDefaultParamDict()
    for name, value in state.items():
      self[name] = value

  def LoadFromFile(self, fname):
    """Loads option data either from a python file (if fname ends in ".py"), or
    a pickled Viz2Params file.
      fname -- name of file from which to load options"""
    values = util.LoadByFileName(fname)
    if isinstance(values, Viz2Params):
      values = values._params
    elif not isinstance(values, dict):
      raise ValueError("Unknown data in file %s" % fname)
    for name, value in values.items():
      self[name] = value

def MakeDefaultParamDict():
  """Create a default set of parameters."""
  return dict(
    retina_bias = 1.0,
    retina_enabled = True,
    retina_kwidth = 15,

    s1_bias = 1.0,
    s1_beta = 1.0,
    s1_kwidth = 11,
    s1_num_orientations = 8,
    s1_num_phases = 2,
    s1_scaling = 2,
    s1_shift_orientations = True,

    c1_kwidth = 5,
    c1_scaling = 2,
    c1_whiten = True,

    s2_beta = 5.0,
    s2_bias = 0.1,  # Configured to match distribution of C1 norm under
                    # whitening.
    s2_kwidth = 7,
    s2_scaling = 2,

    c2_kwidth = 3,
    c2_scaling = 2,

    num_scales = 4,
  )
