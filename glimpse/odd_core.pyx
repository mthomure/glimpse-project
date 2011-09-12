cimport cython
cimport numpy as np
import numpy as np

ctypedef np.float32_t act_t

def UpdateWeights(np.ndarray[act_t, ndim = 4] weights, 
    np.ndarray[act_t, ndim = 3] idata, int step):
  """Accumulate co-occurrence statistics for windows of activity sampled from 
  the input maps. The weights for different orientations are learn 
  independently. The weight matrix for a given orientation is only updated when 
  the center unit is "on", which is defined as having positive activation.
  weights -- (4-D array) 3-D weight matrix for each Gabor orientation.
  idata -- (3-D array) C1 or (phase pooled) S1 activity maps, with orientation 
           bands in the *most-significant* position.
  step -- subsampling factor
  """
  # Use type annotations to produce more efficient C code.
  cdef int b, y, x
  cdef int nbands = idata.shape[0]
  cdef int height = idata.shape[1]
  cdef int width = idata.shape[2]
  assert weights.shape[0] == nbands
  assert weights.shape[1] == nbands
  cdef int kw = weights.shape[2]
  assert kw == weights.shape[3]
  cdef int hw = kw / 2
  cdef int fy, fx, fb
  cdef act_t a, i
  # Process each sampled patch of input activity.
  for fb in range(nbands):
    for fy in range(hw, height - hw, step):
      for fx in range(hw, width - hw, step):
        a = idata[fb, fy, fx]
        # Compute co-occurrence statistics if the center unit is "on".
        if a <= 0: continue
        for b in range(nbands):
          for y in range(kw):
            for x in range(kw):
              if y == hw and x == hw: continue
              i = idata[b, fy + y - hw, fx + x - hw]
              # Accumulate only non-negative weights. This means we don't learn
              # when a given unit (e.g., at another orientation) is 
              # anti-correlated with the target unit.
              if i <= 0: continue
              weights[fb, b, y, x] += a * i


#cdef extern from "odd_core_methods.h":
 
             
@cython.boundscheck(False)
def UpdateCovariance(np.ndarray[act_t, ndim = 4] cov, 
    np.ndarray[act_t, ndim = 3] idata, np.ndarray[act_t, ndim = 1] means,
    int step):

  for array in (cov, idata, means):
    assert array.flags['C_CONTIGUOUS']
  assert array.flags['WRITEABLE']
      
  print "UpdateCovariance()"
      
  # Use type annotations to produce more efficient C code.
  cdef int b, y, x
  cdef int nbands = idata.shape[0]
  cdef int height = idata.shape[1]
  cdef int width = idata.shape[2]
  assert cov.shape[0] == nbands
  assert cov.shape[1] == nbands
  assert means.shape[0] == nbands
  cdef int kw = cov.shape[2]
  assert kw == cov.shape[3]
  cdef int hw = kw / 2
  cdef int fy, fx, fb
  cdef act_t center_activity, lateral_activity
  # Process each sampled patch of input activity.
  for fb in range(nbands):
    for fy in range(hw, height - hw, step):
      for fx in range(hw, width - hw, step):
        center_activity = idata[fb, fy, fx] - means[fb]
        for b in range(nbands):
          for y in range(kw):
            for x in range(kw):
              lateral_activity = idata[b, fy + y - hw, fx + x - hw] - means[b]
              cov[fb, b, y, x] += center_activity * lateral_activity
  # cdef int num_elements = (height - kw + 1) * (width - kw + 1)
  # for fb in range(nbands):
    # for b in range(nbands):
      # for y in range(kw):
        # for x in range(kw):
          # cov[fb, b, y, x] /= num_elements
  return

def UpdateThresholdedCovariance(np.ndarray[act_t, ndim = 4] cov, 
    np.ndarray[act_t, ndim = 3] idata, np.ndarray[act_t, ndim = 1] means,
    int step, float zero_point, float tau, np.ndarray[int, ndim=1] counts):
  # Use type annotations to produce more efficient C code.
  cdef int b, y, x
  cdef int nbands = idata.shape[0]
  cdef int height = idata.shape[1]
  cdef int width = idata.shape[2]
  assert cov.shape[0] == nbands
  assert cov.shape[1] == nbands
  assert means.shape[0] == nbands
  assert counts.shape[0] == nbands
  cdef int kw = cov.shape[2]
  assert kw == cov.shape[3]
  cdef int hw = kw / 2
  cdef int fy, fx, fb
  cdef act_t center_activity, lateral_activity
  # Process each sampled patch of input activity.
  for fb in range(nbands):
    for fy in range(hw, height - hw, step):
      for fx in range(hw, width - hw, step):
        center_activity = idata[fb, fy, fx] - means[fb]
        if abs(center_activity - zero_point) < tau:
          continue
        counts[fb] += 1
        for b in range(nbands):
          for y in range(kw):
            for x in range(kw):
              lateral_activity = idata[b, fy + y - hw, fx + x - hw] - means[b]
              cov[fb, b, y, x] += center_activity * lateral_activity
  return

def UpdateWeightsWithMax(np.ndarray[act_t, ndim = 4] weights, 
    np.ndarray[act_t, ndim = 3] idata, int step):
  """Same as UpdateWeights(), but only updates the weight matrix for the most
  active (target) orientation in each patch.
  weights -- (4-D array) 3-D weight matrix for each Gabor orientation.
  idata -- (3-D array) C1 or (phase pooled) S1 activity maps, with orientation 
           bands in the *least-significant* position. That is, the shape of the 
           array is different from that expected by UpdateWeights().
  step -- subsampling factor
  """
  cdef int b, y, x
  cdef int height = idata.shape[0]
  cdef int width = idata.shape[1]
  cdef int nbands = idata.shape[2]
  assert weights.shape[0] == nbands
  assert weights.shape[1] == nbands
  cdef int kw = weights.shape[2]
  assert kw == weights.shape[3]
  cdef int hw = kw / 2
  cdef int fy, fx, fb
  cdef act_t a, i, max_band, max_act
  for fy in range(hw, height - hw, step):
    for fx in range(hw, width - hw, step):
      # Compute Gabor orientation with maximal activity.
      max_band = 0
      max_act = -1
      for fb in range(nbands):
        a = idata[fy, fx, fb]
        if a > max_act:
          max_act = a
          max_band = fb
      # Update that orientation's weight matrix if the target unit is "on".
      if max_act <= 0: continue
      for y in range(kw):
        for x in range(kw):
          for b in range(nbands):
            if y == hw and x == hw: continue
            i = idata[fy + y - hw, fx + x - hw, b]
            # As in UpdateWeights(), only detect positively correlated units.
            if i <= 0: continue
            weights[max_band, b, y, x] += max_act * i
  return

def ApplyWeights(np.ndarray[act_t, ndim = 4] weights,
    np.ndarray[act_t, ndim = 3] idata, float tau, int step):
  """Apply a set of ODD kernels at every location in which the conditioning unit
  is "on".
  weights -- array containing one ODD kernel per Gabor orientation
  idata -- activity maps for each Gabor orientation
  tau -- threshold used to determine when a unit is "on"
  step -- how many steps to take (either horizontally or vertically) between 
          kernel applications
  """
  cdef int nbands = idata.shape[0]
  cdef int height = idata.shape[1]
  cdef int width = idata.shape[2]
  assert weights.shape[0] == nbands
  cdef int kw = weights.shape[2]
  assert kw == weights.shape[3]
  cdef int hw = kw / 2
  cdef int fy, fx, fb
  cdef int b, y, x
  cdef act_t v
  # Apply a different ODD kernel for each Gabor orientation.
  for fb in range(nbands):
    # Apply the kernel for orientation (fb) at every location.
    for fy in range(0, height - kw, step):
      for fx in range(0, width - kw, step):
        a = idata[fb, fy, fx]
        # Don't apply the kernel if the target unit is inactive.
        if a <= tau: continue
        # Apply kernel for band (fb) to the activity maps at location (fy, fx).
        for b in range(nbands):
          for y in range(kw):
            for x in range(kw):
              w = weights[fb, b, y, x]
              i = idata[b, fy + y, fx + x]
              w *= i
              # Bound activation values to [0, 1].
              if w > 1:
                w = 1
              idata[b, fy + y, fx + x] = w
  return

