"""Functions for dealing with the kernel matrices of linear filters."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import math
import numpy as np

from .garray import ScaleUnitNorm, fromimage

def MakeGaborKernel(kwidth, theta, gamma = 0.6, sigma = None, phi = 0,
    lambda_ = None, scale_norm = True):
  """Create a kernel matrix by evaluating the 2-D Gabor function.

  :param kwidth: Width of the kernel.
  :type kwidth: odd int
  :param float theta: Orientation of the (normal to the) preferred frequency,
     given in radians.
  :param float gamma: Aspect ratio of Gaussian window (*gamma* = 1 means a
     circular window, while 0 < *gamma* < 1 means the window is elongated).
  :param float sigma: Standard deviation of Gaussian window (1/4 the wavelength
     is a good choice, so that the window covers approximately two full
     wavelengths of the sine function).
  :param float phi: Phase offset (*phi* = 0 detects white edge on black
     background, *phi* = :math:`\pi` detects black edge on white background).
  :param float lambda_: Wavelength of sine function (2/5 * *kwidth* is a good
     choice, so that the kernel can fit 2 1/2 wavelengths total).
  :param bool scale_norm: If true, the kernel vector will be scaled to have unit
     norm.
  :returns: Kernel matrix indexed by y-offset and x-offset.
  :rtype: 2D ndarray of float

  Examples:

  >>> kwidth, theta = 11, math.pi / 4
  >>> kernel = MakeGaborKernel(kwidth, theta)
  >>> assert(kernel.shape == (kwidth, kwidth))

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
  """Create a set of 2-D square kernel arrays, whose components are chosen
  according to the Gabor function.

  This function creates kernels across a range of orientation and phase.

  :param kwidth: Width of kernel.
  :type kwidth: odd int
  :param int num_orientations: Number of edge orientations (orientations will be
     spread between 0 and :math:`\pi`).
  :param int num_phases: Number of Gabor phases (a value of 2 matches a white
     edge on a black background, and vice versa).
  :param bool shift_orientations: Whether to rotate Gabor through a small angle
     (a value of True helps compensate for aliasing).
  :returns: Kernel arrays indexed by orientation and phase.
  :rtype: 4D ndarray of float

  Examples:

  >>> kwidth, num_orientations, num_phases = 11, 8, 2
  >>> kernels = MakeGaborKernels(kwidth, num_orientations, num_phases,
          shift_orientations = True)
  >>> assert(kernels.shape == (num_orientations, num_phases, kwidth, kwidth))

  """
  from math import pi
  if shift_orientations:
    offset = 0.5
  else:
    offset = 0
  thetas = pi / num_orientations * (np.arange(num_orientations) + offset)
  phis = 2 * pi * np.arange(num_phases) / num_phases
  ks = [[ MakeGaborKernel(kwidth, theta, phi = phi, **args)
      for phi in phis ]
    for theta in thetas ]
  ks = np.array(ks)
  return ks

def MakeMultiScaleGaborKernels(kwidth, num_scales, num_orientations, num_phases,
    shift_orientations = True, embed_kwidth = None, **args):
  """Create a set of 2-D square kernel arrays, whose components are chosen
  according to the Gabor function.

  This function creates kernels across a range of scale, orientation, and phase.

  :param kwidth: Width of kernel.
  :type kwidth: odd int
  :param int num_scales: Number of Gabor scales (scales will be spread between
     `kwidth` / 8 and slightly less than `kwidth` / 2).
  :param int num_orientations: Number of edge orientations (orientations will be
     spread between 0 and :math:`\pi`).
  :param int num_phases: Number of Gabor phases (a value of 2 matches a white
     edge on a black background, and vice versa).
  :param bool shift_orientations: Whether to rotate Gabor through a small angle
     (a value of True helps compensate for aliasing).
  :param int embed_kwidth: If desired, the result can be embeded in a wider
     array. For example, this can be useful for debugging when the larger scales
     (governed by `kwidth`) contain clipping artifacts. By default, this is
     equal to `kwidth`.
  :param bool scale_norm: If true, then the kernel vector is scaled to have unit
     norm.
  :returns: Kernel arrays indexed by scale, orientation, and phase.
  :rtype: 5D ndarray of float

  Examples:

  >>> kwidth, num_scales, num_orientations, num_phases = 11, 4, 8, 2
  >>> kernels = MakeMultiScaleGaborKernels(kwidth, num_scales, num_orientations,
          num_phases, shift_orientations = True)
  >>> assert(kernels.shape == (num_scales, num_orientations, num_phases, kwidth,
          kwidth))

  """
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
  ks = [[[ MakeGaborKernel(embed_kwidth, theta, phi = phi, sigma = sigma,
              lambda_ = lambda_, **args)
        for phi in phis ]
      for theta in thetas ]
    for sigma, lambda_ in zip(scales, lambdas) ]
  ks = np.array(ks)
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
  """Draw the line corresponding to a given Gabor.

  The generated line is phase independent.

  :param int norientations: Number of edge orientations.
  :param int kwidth: Width of kernel in pixels.
  :param bool shift_orientations: Whether to rotate Gabor through a small angle.
  :param int line_width: Width of drawn line in pixels.
  :rtype: 2D ndarray of float

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
  data = fromimage(im) / 255
  data = np.rot90(data)
  return data

def MakeRandomKernels(nkernels, kshape, normalize = True, mean = 0,
    std = 0.15):
  """Create a set of N-dimensional kernel arrays whose components are sampled
  independently from the normal distribution.

  :param int nkernels: Number of kernels to create.
  :param kshape: Dimensions of each kernel.
  :type kshape: list of int
  :param bool normalize: Whether the resulting kernels should be scaled to have
     unit norm.
  :param float mean: Center of the component-wise normal distribution.
  :param float std: Standard deviation of the component-wise normal
     distribution.
  :returns: Set of kernel arrays.
  :rtype: N-dimensional ndarray of float, where `N = len(kshape)+1`

  Examples:

  >>> nkernels = 10
  >>> kshape = (8, 5, 5)
  >>> kernels = MakeRandomKernels(nkernels, kshape)
  >>> assert(kernels.shape[0] == nkernels)
  >>> assert(kernels.shape[1:] == kshape)

  """
  shape = (nkernels,) + kshape
  kernels = np.random.normal(mean, std, shape).astype(np.float32)
  if normalize:
    for k in kernels:
      # Scale vector norm of given array to unity.
      k /= np.linalg.norm(k)
  return kernels

def MakeUniformRandomKernels(nkernels, kshape, normalize = True, low = 0, high = 1):
  """Sample kernel arrays from multivariate uniform distribution.

  :param int nkernels: Number of kernels to create.
  :param kshape: Dimensions of each kernel.
  :type kshape: list of int
  :param bool normalize: Whether the resulting kernels should be scaled to have
     unit norm.
  :param float low: Lower bound on sampled values (inclusive).
  :param float high: Upper bound on sampled values (exclusive).
  :returns: Set of kernel arrays.
  :rtype: N-dimensional ndarray of float, where `N = len(kshape)+1`

  Examples:

  >>> nkernels = 10
  >>> kshape = (8, 5, 5)
  >>> kernels = MakeUniformRandomKernels(nkernels, kshape)
  >>> assert(kernels.shape[0] == nkernels)
  >>> assert(kernels.shape[1:] == kshape)

  """
  shape = (nkernels,) + tuple(kshape)
  kernels = np.random.uniform(low, high, shape)
  if normalize:
    for kernel in kernels:
      kernel /= np.linalg.norm(kernel)
  return kernels

def MakeLanczosKernel():
  """Construct a smoothing filter based on the Lanczos window.

  .. seealso::
     For more information image resampling using the Lanczos kernel, see the
     `Wikipedia article`_ or the `StackOverflow discussion`_.

  .. _StackOverflow discussion: http://stackoverflow.com/questions/1854146/
     what-is-the-idea-behind-scaling-an-image-using-lanczos

  .. _Wikipedia article: http://en.wikipedia.org/wiki/Lanczos_resampling

  :returns: Fixed-size, square kernel.
  :rtype: 2D ndarray

  """
  from numpy import arange, sinc, outer
  a = 3.0
  # Use a step size smaller than one to expand the kernel size.
  step = 1
  X = arange(-a, a+1, step)
  k1 = sinc(X) * sinc(X / float(a))
  k2 = outer(k1, k1)
  return k2

def Blackman1d(n, alpha = 0.16):
  """The 1-dimensional Blackman window.

  .. deprecated:: 0.1
     Use :func:`numpy.blackman` instead.

  :param int n: Number of elements.
  :param float alpha: The parameter of the Blackman window.
  :rtype: 1D ndarray

  """
  a0 = (1 - alpha) / 2.0
  a1 = 0.5
  a2 = alpha / 2.0
  x = np.arange(n)
  return a0 - a1 * np.cos(2 * math.pi * x / (n - 1)) + \
      a2 * np.cos(4 * math.pi * x / (n - 1))

def Blackman2d(ny, nx, power = 1):
  """The 2-dimensional Blackman window.

  :param int ny: Number of elements along the Y-axis.
  :param int nx: Number of elements along the X-axis.
  :param float power: Elongates the X-direction (if greater than 1), or shortens
     it (if less than 1).
  :rtype: 2D ndarray

  """
  a = np.empty([ny, nx])
  bx = Blackman1d(nx)
  bx = np.maximum(bx, np.zeros_like(bx)) ** power
  by = Blackman1d(ny)
  a[:] = bx
  a = (a.T * by).T
  return np.maximum(a, np.zeros_like(a))

def Gabor(sigma, theta, phi, gamma, lambda_, kernel):
  """Fill a 2D matrix using values of the Gabor function.

  :param float sigma: Variance of the Gaussian function.
  :param float theta: Orientation of the parallel stripes of the Gabor function.
  :param float phi: Phase offset (match a black edge on on white background, or
     vice versa).
  :param float gamma: Spatial aspect ratio. set to 1.0 to get circular variance
     on the Gaussian function, and set to less than one to get elongated
     variance.
  :param float lambda_: Wavelength of sinusoidal function.
  :param kernel: Array in which to store kernel values (must not be None).
  :type kernel: 2D ndarray of float
  :returns: The *kernel* parameter.
  :rtype: 2D ndarray of float

  """
  height, width = kernel.shape
  size = height * width
  for j in range(height):
    for i in range(width):
      y = j - height / 2.0
      x = i - width / 2.0
      yo = -1.0 * x * math.sin(theta) + y * math.cos(theta)
      xo = x * math.cos(theta) + y * math.sin(theta)
      kernel[j,i] = math.exp(-1.0 *
                      (xo**2 + gamma**2 * yo**2) / (2.0 * sigma**2)) * \
                    math.sin(phi + 2 * math.pi * xo / lambda_)
  return kernel
