# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

# Functions for dealing with kernel matrices of filter operations.

from garray import ScaleUnitNorm, ACTIVATION_DTYPE
from gimage import ImageToArray
import math
import numpy as np

def MakeGaborKernel(kwidth, theta, gamma = 0.6, sigma = None, phi = 0,
    lambda_ = None, scale_norm = True):
  """Create a kernel matrix by evaluating the 2-D Gabor function.
  kwidth - width of the kernel (should be odd)
  theta - orientation of the (normal to the) preferred frequency
  gamma - aspect ratio of Gaussian window (gamma = 1 means a circular window,
          0 < gamma < 1 means the window is elongated)
  sigma - standard deviation of Gaussian window (1/4 wavelength is good choice,
          so that the window covers approximately two full wavelengths of the
          sine function)
  phi - phase offset (phi = 0 detects white edge on black background, phi = PI
        detects black edge on white background)
  lambda_ - wavelength of sine function (2/5 * kwidth is good choice, so that
            the kernel can fit 2 1/2 wavelengths total)
  scale_norm - if true, then rescale kernel vector to have unit norm
  """
  from numpy import sin, cos, exp, mgrid
  from math import pi
  if lambda_ == None:
    # Allow four cycles of sine wave
    lambda_ = kwidth / 4.0
  if sigma == None:
    # Window should (approximately) go to zero after two wavelengths from center
    sigma = lambda_ / 2.0
  w = kwidth / 2
  Y, X = mgrid[-w:w+1, -w:w+1]
  Yp = -X * sin(theta) + Y * cos(theta)
  Xp = X * cos(theta) + Y * sin(theta)
  k0 = sin(2 * pi * Xp / lambda_ + phi)  # sine wave
  k1 = exp(-(Xp**2 + gamma**2 * Yp**2) / (2.0 * sigma**2))  # Gaussian window
  kernel = k0 * k1  # windowed sine wave
  if scale_norm:
    ScaleUnitNorm(kernel)
  return kernel

def MakeGaborKernels(kwidth, num_orientations, num_phases, shift_orientations,
    **args):
  """Create a set of 2-D square kernel arrays whose components are chosen
  according to the Gabor function.
  kwidth - width of kernel (should be odd)
  num_orientations - number of edge orientations (orientations will be spread
                     between 0 and pi)
  num_phases - number of Gabor phases (a value of 2 matches a white edge on a
               black background, and vice versa)
  shift_orientations - whether to rotate Gabor through a small angle (a value of
                       True helps compensate for aliasing)
  scale_norm - if true, then rescale kernel vector to have unit norm
  RETURNS: 4-D array of kernel values
  """
  from math import pi
  if shift_orientations:
    offset = 0.5
  else:
    offset = 0
  thetas = pi / num_orientations * (np.arange(num_orientations) + offset)
  phis = 2 * pi * np.arange(num_phases) / num_phases
  ks = np.array([[ MakeGaborKernel(kwidth, theta, phi = phi, **args)
      for phi in phis ] for theta in thetas ], ACTIVATION_DTYPE)
  return ks

def MakeMultiScaleGaborKernels(kwidth, num_scales, num_orientations, num_phases,
    shift_orientations = True, embed_kwidth = None, **args):
  from math import pi
  if shift_orientations:
    offset = 0.5
  else:
    offset = 0
  if embed_kwidth == None:
    embed_kwidth = kwidth
  fscale = np.arange(num_scales) / float(num_scales)
  scales = (kwidth / 4.0) * (0.5 + 1.5 * fscale)
  lambdas = 2 * scales
  thetas = pi / num_orientations * (np.arange(num_orientations) + offset)
  phis = 2 * pi * np.arange(num_phases) / num_phases
  ks = np.array([[[
          MakeGaborKernel(embed_kwidth, theta, phi = phi, sigma = sigma,
              lambda_ = lambda_, **args)
        for phi in phis ]
      for theta in thetas ]
    for sigma, lambda_ in zip(scales, lambdas) ], ACTIVATION_DTYPE)
  return ks

# gaussian scale is:
#   sigmas = 11/32 * arange(4, 14, 3)
#          = [ 44/32, 77/32, 110/32, 143/32 ]
#          = [ 1.4, 2.4, 3.4, 4.5 ]
# sinusoidal wavelength is:
#   lambdas = sigmas * 2
#   lambdas = 11/16 * arange(4, 14, 3) => [ 2.8, 4.8, 6.9, 8.9 ]
# parameterized response frequencies:
#   fs = 1/lambdas = [ 0.36, 0.21, 0.15, 0.11 ]
# equivalent down-sampling rates are:
#   fs / fs[0] = [ 1.0, 0.57142857, 0.4, 0.30769231 ]

# measured response frequencies (via power spectrum):
#   f_0 = [ 0.357 / 0.162, 0.357 / 0.170 ]
#   f_1 = [ 0.197 / 0.145, 0.209 / 0.180 ]
#   f_2 = [ 0.135 / 0.121, 0.154 / 0.135 ]
#   f_3 = [ 0.117 / 0.109, 0.117 / 0.111 ]

# measured wavelength(via power spectrum):
#   lambda_0 = [ 2.798 / 0.162, 2.798 / 0.170 ]
#   lambda_1 = [ 4.785 / 0.180, 5.069 / 0.145 ]
#   lambda_2 = [ 6.481 / 0.135, 7.420 / 0.121 ]
#   lambda_3 = [ 8.533 / 0.109, 8.533 / 0.111 ]

def DrawGaborAsLine(orient, num_orientations = 8, kwidth = 11,
    shift_orientations = False, line_width = 1):
  """Draw the line corresponding to a given Gabor. The generated line is phase
  independent.
  norientations - number of edge orientations
  kwidth - width of kernel in pixels
  shift_orientations - whether to rotate Gabor through a small angle
  line_width - width of drawn line in pixels
  """
  from math import pi, tan
  assert orient < num_orientations, "Expected orientation in [0, %s]: got %s" %\
      (num_orientations, orient)
  if shift_orientations:
    theta_shift = 0.5
  else:
    theta_shift = 0
  theta = pi / num_orientations * (theta_shift + orient)
  hw = kwidth / 2
  theta_p = theta % pi
  if theta_p < pi / 4 or theta_p > 3 * pi / 4:
    # compute y from x
    x1 = -hw
    x2 = hw
    y1 = tan(theta) * x1
    y2 = tan(theta) * x2
  else:
    # compute x from y
    y1 = -hw
    y2 = hw
    x1 = y1 / tan(theta)
    x2 = y2 / tan(theta)
  im = Image.new("L", (kwidth, kwidth), 0)
  draw = ImageDraw.Draw(im)
  draw.line((x1 + hw, y1 + hw, x2 + hw, y2 + hw), fill = 255,
      width = line_width)
  data = ImageToArray(im) / 255
  data = np.rot90(data)
  return data

def MakeRandomKernels(nkernels, kshape, normalize = True, mean = 0,
    std = 0.15):
  """Create a set of N-dimensional kernel arrays whose components are sampled
  independently from the normal distribution.
  nkernels - number of kernels to create
  kshape - dimensions of each kernel
  normalize - whether the resulting kernels should be scaled to have unit norm
  mean - center of the component-wise normal distribution
  std - standard deviation of the component-wise normal distribution
  RETURNS: N-D array of kernel values [where N = len(kshape)+1]
  """
  shape = (nkernels,) + kshape
  kernels = np.random.normal(mean, std, shape).astype(np.float32)
  if normalize:
    for k in kernels:
      # Scale vector norm of given array to unity.
      k /= np.linalg.norm(k)
  return kernels

def MakeLanczosKernel():
  """Construct a smoothing filter based on the Lanczos window.
  RETURNS (2-d array) fixed-size, square kernel
  http://en.wikipedia.org/wiki/Lanczos_resampling
  http://stackoverflow.com/questions/1854146/what-is-the-idea-behind-scaling-an-
  image-using-lanczos
  """
  from numpy import arange, sinc, outer
  a = 3.0
  # Use a step size smaller than one to expand the kernel size.
  step = 1
  X = arange(-a, a+1, step)
  k1 = sinc(X) * sinc(X / float(a))
  k2 = outer(k1, k1)
  return k2
