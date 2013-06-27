"""Miscellaneous functions related to images and image processing."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import Image
import numpy as np
from scipy import fftpack
import sys
import warnings

from .garray import fromimage, toimage

def ScaleImage(img, size):
  """Resize an image.

  :param Image img: Input image.
  :param size: Size of output image in the format (width, height).
  :type size: 1D array-like of float or int
  :return: Resized version of input image.
  :rtype: Image

  """
  size = np.array(size, dtype = int)
  # Use bicubic interpolation if the new width is larger than the old width.
  if size[0] > img.size[0]:
    method = Image.BICUBIC  # interpolate
  else:
    method = Image.ANTIALIAS  # blur and down-sample
  return img.resize(size, method)

def ScaleAndCropImage(img, size):
  """Resize an image by scaling and cropping.

  Input is resized to match `size` by 1) scaling the short edge while preserving
  aspect ratio, and 2) removing extra border pixels from the long edge.

  :param Image img: Input image.
  :param size: Size of output image in the format (width, height).
  :type size: 1D array-like of float or int
  :return: Resized version of input image.
  :rtype: Image

  """
  size = np.array(size, dtype = int)
  img_width, img_height = img.size
  image_rho = img_width / float(img_height)  # aspect ratio of input
  target_width, target_height = size
  target_rho = target_width / float(target_height)  # aspect ratio of output
  if image_rho > target_rho:  # crop left and right borders
    # Scale to target height (maintaining aspect ratio) and crop border pixels
    # from left and right edges. Note that the scaled width is guaranteed to
    # be at least as large as the target width.
    scaled_width = int(float(target_height) * image_rho)
    img = ScaleImage(img, size = (scaled_width, target_height))
    border = int((scaled_width - target_width) / 2.)
    # Bounding box format is left, upper, right, and lower; where the point
    # (0,0) corresponds to the top-left corner of the image.
    img = img.crop(box = (border, 0, border + target_width, target_height))
  else:  # crop top and bottom borders
    # Scale to target width (maintaining aspect ratio) and crop border pixels
    # from top and bottom edges. Note that the scaled height is guaranteed to
    # be at least as large as the target height.
    scaled_height = int(float(target_width) / image_rho)
    img = ScaleImage(img, size = (target_width, scaled_height))
    border = int((scaled_height - target_height) / 2.)
    # Bounding box format is left, upper, right, and lower; where the point
    # (0,0) corresponds to the top-left corner of the image.
    img = img.crop(box = (0, border, target_width, border + target_height))
  assert np.all(img.size == size), ("Result image size is %s, " % img.size) + \
      "but requested %s" % size
  return img

def PowerSpectrum2d(image, width = None):
  """Compute the 2-D power spectrum for an image.

  :param image: Image data.
  :type image: 2D ndarray
  :param int width: Width of image to use for FFT (i.e., image width plus
     padding). By default, this is the width of the image.
  :returns: Squared amplitude from FFT of image.
  :rtype: 2D ndarray

  """
  if width != None:
    image = PadArray(image,
        np.repeat(width, 2),  # shape of padded array
        0)  # border value
  from scipy.fftpack import fftshift, fft2
  return np.abs(fftshift(fft2(image))) ** 2

def PowerSpectrum(image, width = None):
  """Get the 1-D power spectrum (squared-amplitude at each frequency) for a
  given input image. This is computed from the 2-D power spectrum via a
  rotational average.

  :param image: Image data.
  :type image: 2D ndarray
  :param int width: Width of image to use for FFT (i.e., image width plus
     padding). By default, this is the width of the image.
  :returns: Array whose rows contain the value, sum, and count of bins in the
     power histogram.

  """
  # from: http://www.astrobetter.com/fourier-transforms-of-images-in-python/
  assert image.ndim == 2
  f2d = PowerSpectrum2d(image, width)
  # Get sorted radii.
  x, y = np.indices(f2d.shape)
  center_x = (x.max() - x.min()) / 2.0
  center_y = (y.max() - y.min()) / 2.0
  r = np.hypot(x - center_x, y - center_y)
  ind = np.argsort(r.flat)
  r_sorted = r.flat[ind]
  # Bin the radii based on integer values. First, find the location (offset) for
  # the edge of each bin.
  r_int = r_sorted.astype(int)
  delta_r = r_int[1:] - r_int[:-1]
  r_ind = np.where(delta_r)[0]
  # Compute the number of elements in each bin.
  size_per_bin = r_ind[1:] - r_ind[:-1]
  # Finally, compute the average value for each bin.
  f_sorted = f2d.flat[ind]
  f_cumsum = np.cumsum(f_sorted, dtype = float)  # total cumulative sum
  sum_per_bin = f_cumsum[r_ind[1:]] - f_cumsum[r_ind[:-1]]  # cum. sum per bin
  # Use a circular window
  size = min(f2d.shape)
  sum_per_bin = sum_per_bin[: size / 2]
  size_per_bin = size_per_bin[: size / 2]
  # Compute the frequency (in cycles per pixel) corresponding to each bin.
  freq = np.arange(0, size / 2).astype(float) / size
  # Compute the average power for each bin.
  # XXX the average may be significantly more accurate than the sum, as there
  # are many fewer low-frequency locations in the FFT.
  #~ avg_per_bin = sum_per_bin / size_per_bin
  return np.array([freq, sum_per_bin, size_per_bin])

def MakeScalePyramid(data, num_layers, scale_factor):
  """Create a pyramid of resized copies of a 2D array.

  :param data: Base layer of the scale pyramid (i.e., highest frequency data).
  :type data: 2D ndarray of float
  :param int num_layers: Total number of layers in the pyramid, including
     the first layer passed as an argument.
  :param float scale_factor: Down-sampling factor between layers. Must be less
     than 1.
  :return: All layers of the scale pyramid.
  :rtype: list of 2D ndarray of float

  """
  if scale_factor >= 1:
    raise ValueError("Scale factor must be less than one.")
  pyramid = [ data ]
  image = toimage(data, mode = 'F')
  for i in range(num_layers - 1):
    size = np.array(image.size, np.int) * scale_factor
    image = image.resize(np.round(size).astype(int), Image.ANTIALIAS)
    pyramid.append(fromimage(image))
  return pyramid

def MakeOneOverFNoise(dim, beta):
  """Generate 1/f spatial noise.

  This function generates 1/f spatial noise, with a normal error
  distribution (the grid must be at least 10x10 for the errors to be normal).
  1/f noise is scale invariant, there is no spatial scale for which the
  variance plateaus out, so the process is non-stationary.

  :param dim: The size of the spatial pattern. For example, dim=[10,5] is a 10x5
     spatial grid.
  :type dim: 2-tuple of int
  :param float beta: The spectral distribution. Spectral density S(f) = N f^BETA
           (f is the frequency, N is normalization coefficient).
                BETA = 0 is random white noise.
                BETA  -1 is pink noise
                BETA = -2 is Brownian noise
           The fractal dimension is related to BETA by, D = (6+BETA)/2

  Note that the spatial pattern is periodic.  If this is not wanted the
  grid size should be doubled and only the first quadrant used.

  Time series can be generated by setting one component of DIM to 1

  The method is briefly described in Lennon, J.L. "Red-shifts and red
  herrings in geographical ecology", Ecography, Vol. 23, p101-113 (2000).

  Many natural systems look very similar to 1/f processes, so generating
  1/f noise is a useful null model for natural systems.

  The errors are normally distributed because of the central
  limit theorem.  The phases of each frequency component are randomly
  assigned with a uniform distribution from 0 to 2*pi. By summing up the
  frequency components the error distribution approaches a normal
  distribution.

  Written by Jon Yearsley  1 May 2004
      j.yearsley@macaulay.ac.uk

  Ported to Python by Mick Thomure  12 December 2012
      thomure@cs.pdx.edu

  """
  from numpy import cos, sin, pi, arange, floor, ceil
  from numpy.matlib import repmat
  from scipy.fftpack import ifft2
  dim = np.array(dim, dtype = float)
  # Generate the grid of frequencies. u is the set of frequencies along the
  # first dimension. First quadrant are positive frequencies.  Zero frequency is
  # at u(1,1).
  u = np.concatenate((arange(floor(dim[0] / 2) + 1),
      -arange(ceil(dim[0] / 2) - 1, 0, -1)))
  u = u.reshape(len(u), 1)
  u /= dim[0]
  # Reproduce these frequencies along ever row.
  u = repmat(u, 1, dim[1])
  # `v` is the set of frequencies along the second dimension.  For a square
  # region it will be the transpose of `u`.
  v = np.concatenate((arange(floor(dim[1] / 2) + 1),
      -arange(ceil(dim[1] / 2) - 1, 0, -1)))
  v = v.reshape(1, len(v))
  v /= dim[1]
  # Reproduce these frequencies along ever column.
  v = repmat(v, dim[0], 1)
  # Generate the power spectrum.
  # This causes intermittent RuntimeWarning: "divide by zero encountered in
  # reciprocal", which we ignore in the proceeding line.
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    S_f = (u ** 2 + v ** 2) ** (beta / 2.)
  # Set any infinities to zero.
  S_f[ S_f == np.inf ] = 0
  # Generate a grid of random phase shifts.
  phi = np.random.random(np.array(dim, dtype = int))
  # Inverse Fourier transform to obtain the the spatial pattern.
  x = ifft2(S_f ** .5 * (cos(2 * pi * phi) + 1j * sin(2 * pi * phi)))
  # Pick just the real component.
  return np.real(x)

def TileImage(data, patch_size):
  """Split a single image into 2D set of 2D image tiles."""
  assert data.ndim == 2
  height, width = data.shape
  height -= patch_size - 1  # avoid incomplete patches near border
  width -= patch_size - 1
  ys, xs = np.mgrid[0:height:patch_size, 0:width:patch_size]
  patches = np.array([ data[y:y+patch_size, x:x+patch_size]
      for y, x in zip(ys.flat, xs.flat) ])
  patches = patches.reshape(ys.shape + patches.shape[-2:])
  return patches

def UntileImage(patches):
  """Concatenate a 2D set of 2D image tiles."""
  assert patches.ndim == 4
  assert patches.shape[-1] == patches.shape[-2], "Patches must be square"
  height, width, patch_size = patches.shape[:3]
  height *= patch_size
  width *= patch_size
  data = np.ones((height, width), dtype = np.float)
  for y0, x0 in np.ndindex(patches.shape[:2]):
    y = y0 * patch_size
    x = x0 * patch_size
    data[y:y+patch_size, x:x+patch_size] = patches[y0, x0]
  return data
