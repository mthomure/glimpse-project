from glimpse.util import gimage
import numpy as np

#def EmbedKernel(k, embed_width):
  #kh, kw = k.shape[-2:]
  ##~ assert kh % 2 == 1 and kw % 2 == 1, "Kernel width must be odd"
  #embed = np.zeros(k.shape[:-2] + (embed_width, embed_width))
  #khh = kh / 2
  #khw = kw / 2
  #ehw = embed_width / 2
  #a = ehw - khh
  #b = ehw + (kh - khh)
  #c = ehw - khw
  #d = ehw + (kw - khw)
  #embed[a:b, c:d] = k
  #return embed

#def KernelPowerSpectrum(k, embed_width = 512):
  #ek = EmbedKernel(k, embed_width)
  #freqs, sums, cnts = gimage.PowerSpectrum(ek)
  #return freqs, sums

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

#def ShowKernelPowerSpectrum(k, embed_width = 512, save = False, **args):
  #from matplotlib import pyplot
  #assert len(k.shape) == 2
  #kwidth = k.shape[-1]
  #freqs, power = KernelPowerSpectrum(k, embed_width)
  ##~ pyplot.figure(2)
  ##~ pyplot.clf()
  #pyplot.plot(freqs, power, **args)
  #pyplot.yticks([])
  #pyplot.xlabel('Cycles per Pixel')
  #pyplot.ylabel('Power')
  ##~ pyplot.suptitle('Example Power Spectrum for %dx%d Kernels%s' % (kwidth,
      ##~ kwidth, title))
  #if save:
    #pyplot.savefig('%df.png' % kwidth)

def KernelFFT(k, embed_width = 512):
  return gimage.PowerSpectrum2d(EmbedKernel(k, embed_width))

def ShowKernelFFT(k, embed_width = 512, save = False):
  from glimpse.util import gplot
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

def MakeSinusoid2d(kwidth = 11, freq = 0.25, theta = 0, phi = 0):
  from math import pi
  from numpy import mgrid, sin, cos
  #~ assert kwidth % 2 == 1
  w = kwidth / 2
  lambda_ = 1.0 / freq
  Y, X = mgrid[0:kwidth, 0:kwidth] #-w:w+1, -w:w+1]
  Xp = X * cos(theta) + Y * sin(theta)
  return sin(2 * pi * Xp / lambda_ + phi)  # sine wave
