# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

"""Functions for summarizing matrix datasets with repeated observations."""
import numpy as np

def xs(data):
  return np.unique(data[:,0])

def ys(data):
  return [ data[data[:,0] == x][:,1] for x in np.unique(data[:,0]) ]

def ys_mean(data):
  return map(np.mean, ys(data))

def ys_std(data):
  return map(np.std, ys(data))

def ys_stderr(data):
  return [ np.std(y) / len(y) for y in ys(data) ]
