# Ipython (pylab) script for playing with the ideas in Saxe et al (2010).

#gray()

# make random patch
x = np.random.normal(size=(32, 32))
x -= x.mean()
#figure().suptitle('patch')
#imshow(x)

# look at fft
y = fftshift(fft2(x))
#figure().suptitle('fft')
#imshow( log(abs(y)) )

# see that components work
yr = real(y)
yi = imag(y)
#figure().suptitle('component reconstruction')
#imshow(real(ifft2(fftshift(yr + yi * 1j))))

# block out all other frequencies
m = np.where(yr == yr.max())
yr2 = zeros(yr.shape)
yr2[ m[0], m[1] ] = 1
yi2 = zeros(yi.shape)
main_component = real(ifft2(fftshift(yr2 + yi2 * 1j)))
main_component *= x.max() / main_component.max()
#figure().suptitle('blocked reconstruction')
#imshow(main_component)

import ImageDraw
def DrawLocations(idata, locs):
  img = util.ArrayToGreyscaleImage(idata).convert("RGB")
  w, h = img.size
  kw = 2
  red = (255, 0, 0)
  draw = ImageDraw.Draw(img)
  for y, x in locs:
    x2 = x
    # flip the y-axis
    y2 = h - y
    bbox = (x2 - kw, y2 - kw, x2 + kw, y2 + kw)
    draw.ellipse(bbox, outline = red)
  return img

test = np.zeros((96, 160))
test[32:64, 32:64] = x
test[32:64, 96:128] = main_component
#figure().suptitle('test image')
#imshow(test)

from scipy.ndimage import filters
kwidth = 8
o = filters.correlate(test, x)
figure().suptitle('square pooling')
t = filters.correlate(o**2, np.ones((kwidth, kwidth)) / (kwidth * kwidth))
img = DrawLocations(t, np.argwhere(t == t.max()))
imshow(img)

figure().suptitle('max pooling')
t = filters.maximum_filter(o, (kwidth, kwidth))
img = DrawLocations(t, [ np.argwhere(t == t.max())[0] ])
imshow(img)

