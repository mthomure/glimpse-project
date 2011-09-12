import scipy

# Reconstruction of layer activity (e.g., the input image) from higher layer 
# activity maps (e.g., S1 activation).

def reconstruct_from_s1(results, power):
  # assumes no retinal layer, and no down-sampling at s1.
  # assumes 4 scales
  s1k = smoosh(results.s1_kernels.copy())
  
  # Weight kernels by their L1 norm
  for k in s1k:
    k[:] /= norm(k.flat, power)
  
  # half-width of the s1 kernel
  hw = results.options['s1_kwidth'] / 2
  # full height and width of original image
  full_shape = results.r_activity[0].shape
  num_scales = results.options['num_scales']
  # create per-scale reconstructions
  recons = np.zeros((num_scales,) + full_shape)
  recons2 = []
  for scale in range(num_scales):
    s1 = smoosh(results.s1_activity[scale])
    # create per-band reconstructions
    recon_per_band = np.array([ filters.convolve(s, k) for s, k in
        zip(s1, s1k) ], np.float32)
    img = recon_per_band.sum(0)
    h, w = img.shape
    img2 = np.zeros_like(results.r_activity[scale])
    # account for padding
    img2[hw : hw + h, hw : hw + w ] = img

    if scale == 0:
      recons[scale] = img2
    else:
      # resize array
      fh, fw = full_shape
  
      img2b = scipy.ndimage.interpolation.zoom(img2, (fh / float(h), 
          fw / float(w)))
      h2, w2 = img2b.shape
      # z = np.zeros(full_shape, img2b.dtype)
      # z[0:h2, 0:w2] = img2b
      # recons2.append(z)
  
      print img2b.shape, h2, w2
  
      recons[scale, 0:h2, 0:w2] = img2b[:]

    # img2 = util.ArrayToGreyscaleImage(img2).resize(full_shape)
    # recons[scale] = util.ImageToArray(img2)
  # combine per-scale reconstructions
  return recons.sum(0), recons, s1k

