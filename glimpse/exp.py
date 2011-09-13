from glimpse import core, util
from glimpse.util import image
from glimpse.core import c_src, misc
import numpy as np
import math

def To3d(array):
  return array.reshape((-1,) + array.shape[-2:])

def EmbedKernel(k, embed_width):
  kh, kw = k.shape[-2:]
  #~ assert kh % 2 == 1 and kw % 2 == 1, "Kernel width must be odd"
  embed = np.zeros(k.shape[:-2] + (embed_width, embed_width))
  khh = kh / 2
  khw = kw / 2
  ehw = embed_width / 2
  a = ehw - khh
  b = ehw + (kh - khh)
  c = ehw - khw
  d = ehw + (kw - khw)
  embed[a:b, c:d] = k
  return embed

def KernelPowerSpectrum(k, embed_width = 512):
  ek = EmbedKernel(k, embed_width)
  freqs, sums, cnts = image.PowerSpectrum(ek)
  return freqs, sums

def FitGaussian(y, x = None):
  """Treat a list of values (y) as the probabilities of thier integer indices (x)."""
  assert len(data.shape) == 1
  from numpy import arange, sum, abs, sqrt, exp
  probs = data / data.sum()
  indices = arange(probs.size)
  mean = sum(probs * indices)
  std = sqrt(abs(sum(
      (indices - mean)**2 * probs
      )))
#  max_val = data.max()
#  return max_val * exp(-(indices - mean)**2 / (2 * std**2))
  return mean, std

def FitPowerSpectrumToGaussian(freqs, power):
  mean, std = FitGaussian(power)
  mean_freq = freqs[ round(mean) ]
  std_freq = freqs[ round(mean + std) ] - mean_freq
  return mean_freq, std_freq

def ShowKernelPowerSpectrum(k, embed_width = 512, save = False):
  from matplotlib import pyplot
  assert len(k.shape) == 2
  kwidth = k.shape[-1]
  freqs, power = KernelPowerSpectrum(k, embed_width)
  #~ pyplot.figure(2)
  #~ pyplot.clf()
  pyplot.plot(freqs, power)
  pyplot.yticks([])
  pyplot.xlabel('Cycles per Pixel')
  pyplot.ylabel('Power')
  #~ pyplot.suptitle('Example Power Spectrum for %dx%d Kernels%s' % (kwidth,
      #~ kwidth, title))
  if save:
    pyplot.savefig('%df.png' % kwidth)

def KernelFFT(k, embed_width = 512):
  return image.PowerSpectrum2d(EmbedKernel(k, embed_width))

def ShowKernelFFT(k, embed_width = 512, save = False):
  from glimpse.util import plot as gplot
  from matplotlib import pyplot
  # Show the 2D FFT of the kernel
  f = KernelFFT(k, embed_width)
  #~ pyplot.figure(f1idx)
  #~ pyplot.clf()
  gplot.Show2DArray(f)
  #~ pyplot.suptitle('Example FFT for kwidth = %d%s' % (kwidth, title))
  if save:
    pyplot.savefig('%dt.png' % kwidth)

def KernelPowerStats(k):
  """Get the mean and std dev of power spectrum."""
  freqs, power = KernelPowerSpectrum(k, 512)
  mean_idx = power.argmax()
  mean_freq = freqs[ mean_idx ]
  std_idx = np.where(power[:mean_idx] < power[mean_idx] / 3)[0].max()
  std_freq = mean_freq - freqs[std_idx]
  return mean_freq, std_freq

#~ def MakeSinusiod2d(kwidth):
  #~ from numpy import mgrid, sin, cos
  #~ from math import pi
  #~ assert kwidth % 2 == 1
  #~ w = kwidth / 2
  #~ theta = pi / 8
  #~ lambda_ = kwidth / 8.0
  #~ phi = 0
  #~ Y, X = mgrid[-w:w+1, -w:w+1]
  #~ Yp = -X * sin(theta) + Y * cos(theta)
  #~ Xp = X * cos(theta) + Y * sin(theta)
  #~ k0 = sin(2 * pi * Xp / lambda_ + phi)  # sine wave
  #~ return k0

def MakeLanczosKernel(): #kwidth):
  """Construct a smoothing filter based on the Lanczos window.
  http://en.wikipedia.org/wiki/Lanczos_resampling
  http://stackoverflow.com/questions/1854146/what-is-the-idea-behind-scaling-an-image-using-lanczos
  """
#  assert kwidth % 2 == 1
#  a = kwidth / 2
  from numpy import arange, sinc, outer
  a = 3.0
  X = arange(-a, a+1)
  k1 = sinc(X) * sinc(X / float(a))
  k2 = outer(k1, k1)
  return k2

def MakeLanczosKernel1d(factor, support):
  from numpy import mgrid, sinc
  from numpy import sin
  from math import pi
  kwidth = int(factor * support + 0.5)
  # Generate inputs
  X = mgrid[ -kwidth : kwidth+1 ].astype(np.float) / factor
  # Apply Lanczos function
  Y = sinc(X) * sinc(X / support)
  Y[len(Y)/2] = 1
  return Y

def MakeDisc(kwidth):
  from numpy import linspace, meshgrid
  Z = linspace(-1, 1, kwidth)
  X, Y = meshgrid(Z, Z)
  return X**2 + Y**2 <= 1

def MakeSinusiod2d(kwidth = 11, freq = 0.25, theta = 0, phi = 0):
  from math import pi
  from numpy import mgrid, sin, cos
  #~ assert kwidth % 2 == 1
  w = kwidth / 2
  lambda_ = 1.0 / freq
  Y, X = mgrid[0:kwidth, 0:kwidth] #-w:w+1, -w:w+1]
  Xp = X * cos(theta) + Y * sin(theta)
  return sin(2 * pi * Xp / lambda_ + phi)  # sine wave


def MakeMultiscaleGaborKernels(kwidth, num_scales, num_orientations, num_phases,
    shift_orientations, **args):
  from math import pi
  if shift_orientations:
    offset = 0.5
  else:
    offset = 0
  fscale = np.arange(num_scales) / float(num_scales)
  scales = (kwidth / 4.0) * (0.5 + 1.5 * fscale)
  lambdas = 2 * scales
  thetas = pi / num_orientations * (np.arange(num_orientations) + offset)
  phis = 2 * pi * np.arange(num_phases) / num_phases
  ks = np.array([[[
          misc.MakeGaborKernel(kwidth, theta, phi = phi, sigma = sigma,
              lambda_ = lambda_, **args)
        for phi in phis ]
      for theta in thetas ]
    for sigma, lambda_ in zip(scales, lambdas) ], c_src.activation_dtype)
  return ks

# gaussian scale is:
#   sigmas = 11/32 * arange(4, 14, 3) => [ 44/32, 77/32, 110/32, 143/32 ] = [ 1.4, 2.4, 3.4, 4.5 ]
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
