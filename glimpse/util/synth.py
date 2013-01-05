"""Utilities for generating synthetic image datasets.

"""

import os
import Image
import numpy as np
import math
import glob
import sys
from decorator import decorator  # need "decorator" from PyPi
from tempfile import NamedTemporaryFile
import subprocess

from glimpse.util.gimage import fromimage, toimage


def debug(*names, **args):
  for n in names:
    print "%s:" % n, args.get(n, None)

import logging, time
def log(*args):
  #~ logging.warn("%s", map(str, args))
  print " ".join(map(str, args))
  sys.stdout.flush()
  #~ time.sleep(1)

def _memoize(func, *args, **kw):
  if kw: # frozenset is used to ensure hashability
    key = args, frozenset(kw.iteritems())
  else:
    key = args
  cache = func.cache # attributed added by memoize
  if key in cache:
    return cache[key]
  else:
    cache[key] = result = func(*args, **kw)
    return result

def memoize(f):
  """A function decorator that caches prior results."""
  f.cache = {}
  return decorator(_memoize, f)

class SingletonDict(dict):
  """A dictionary with at most one item."""

  def __setitem__(self, key, value):
    if key not in self:
      self.clear()
      return super(SingletonDict, self).__setitem__(key, value)

def cache_last(f):
  """A function decorator that caches the last result."""
  f.cache = SingletonDict()
  return decorator(_memoize, f)

@decorator
def trace(f, *args, **kw):
  print "calling %s with args %s, %s" % (f.func_name, args, kw)
  return f(*args, **kw)


class Range(object):
  """A non-empty, half-open interval [low, high) on the number line."""

  def __init__(self, low, high, ):
    assert high >= low
    self.low = low
    self.high = high

  def __str__(self):
    return ObjToStr(self, 'low', 'high')

  __repr__ = __str__

  def __contains__(self, value):
    return value >= self.low and value < self.high

  def __mul__(self, value):
    low = self.low * value
    if isinstance(self.low, int):
      low = int(low)
    high = self.high * value
    if isinstance(self.high, int):
      high = int(high)
    return Range(low, high)

  def Sample(self):
    """Draw a sample from the uniform range.

    :rtype: float

    """
    # Note: np.random.random() samples from half-open interval [a, b)
    value = (self.high - self.low) * np.random.random() + self.low
    return value

class ModRange(object):

  def __init__(self, low, high, mod = None):
    """

    :param int low:
    :param int high:
    :param int mod:

    """
    self.low = low % mod
    self.high = high % mod
    self.mod = mod

  def __contains__(self, value):
    value = value % self.mod
    if self.low < self.high:
      return value >= self.low and value < self.high  # in interval
    return value >= self.low or value < self.high  # not in complement

  def __mul__(self, value):
    value = int(value)
    low = (self.low * value) % self.mod
    high = (self.high * value) % self.mod
    return ModRange(low, high, self.mod)

  def Sample(self):
    # Note: np.random.random() samples from half-open interval [a, b)
    size = abs(self.high - self.low)
    if self.low > self.high:
      size = self.mod - size
    value = size * np.random.random() + self.low
    return int(value) % self.mod

  def __str__(self):
    return ObjToStr(self, 'low', 'high', 'mod')

  __repr__ = __str__

class SingletonRange(object):

  def __init__(self, value):
    self.value = value

  def __contains__(self, value):
    return value == self.value

  def __mul__(self, value):
    return SingletonRange(value * self.value)

  def Sample(self):
    return self.value

  def __str__(self):
    return ObjToStr(self, 'value')


def MakeAloiMaskedFg(root, object_id, rotation):
  """Combine ALOI object and mask images."""
  path = os.path.join(str(object_id), "%s_r%s.png" % (object_id, rotation))
  obj_img = Image.open(os.path.join(root, "grey", path))
  mask_img = Image.open(os.path.join(root, "mask", path))
  mask = fromimage(mask_img)
  xidx = np.where(mask.max(0) > 0)[0]  # find columns with non-zero values.
  yidx = np.where(mask.max(1) > 0)[0]  # find rows with non-zero values.
  # find range of non-zero rows and columns as (left, upper, right, lower)
  bbox = (xidx.min(), yidx.min(), xidx.max(), yidx.max())
  obj_img = obj_img.crop(bbox)
  mask_img = mask_img.crop(bbox)
  obj_img.putalpha(mask_img)
  return obj_img

class PovrayRenderError(Exception):

  def __init__(self, returncode, cmd, output):
    self.returncode = returncode
    self.cmd = cmd
    self.output = output

  def __str__(self):
    lines = "\n".join(buff.split("\n")[-10:])  # last 10 lines of povray message
    return "Failed to render Povray object model\n\n" + \
        "== Tail of Povray Output ==\n%s" % lines

def MakePovrayMaskedFg(root, object_id, rotation):
  """Render a 3D object model using Povray.

  :param bool verbose: Do not suppress data written to stderr.
  :rtype: Image
  :returns: Rendered image with mode RGBA

  """
  # make temp file
  temp_file = NamedTemporaryFile(suffix = '.png')
  temp_path = temp_file.name
  # render object to temp file
  render_path = os.path.join(root, 'bin', 'render')
  cmd = map(str, (render_path, object_id, rotation, temp_path))
  try:
    subprocess.check_output(cmd, stderr = subprocess.STDOUT)
  except subprocess.CalledProcessError, e:
    raise PovrayRenderError(e.returncode, e.cmd, e.output)
  # load results from disk
  img = Image.open(temp_path)
  assert img.mode == 'RGBA'
  img.load()  # Image data will be removed from disk when this function returns
  data = fromimage(img)
  mask = data[:, :, -1]
  xidx = np.where(mask.max(0) > 0)[0]  # find columns with non-zero values.
  yidx = np.where(mask.max(1) > 0)[0]  # find rows with non-zero values.
  # find range of non-zero rows and columns as (left, upper, right, lower)
  bbox = (xidx.min(), yidx.min(), xidx.max(), yidx.max())
  img = img.crop(bbox)
  return img

def LoadImage(*path_parts):
  """A non-caching image loader."""
  path_parts = map(str, path_parts)
  path = os.path.join(*path_parts)
  return Image.open(path)

@cache_last
def LoadCachedImage(*path_parts):
  """An image loader that remembers the last file from disk."""
  return LoadImage(*path_parts)

def _ComputeIndices(iw, w, dx):
  """Choose image indices to crop or pad input image to given size.

  :param int iw: Input extent.
  :param int w: Target extent.
  :param int dx: Translation in pixels.
  :rtype: 2-tuple of slice
  :returns: Indices for input data and output data.

  """
  offset = int(math.fabs(math.floor((w - iw) / 2.)))
  if iw <= w:
    # pad image width to fit target, and store to center region of output image
    # (translated by dx)
    out_idx = slice(max(0, offset + dx), min(w, offset + iw + dx))
    length = out_idx.stop - out_idx.start
    if out_idx.start <= 0:
      in_idx = slice(iw - length, iw)
    else:
      in_idx = slice(0, length)
  else:
    raise ValueError("Input image must not be larger than background.")
  return in_idx, out_idx

def _MakeObjectImage(img, size, ds, dy, dx):
  """Render a transformed object with a transparent background.

  :param Image img: Image of foreground object, with an alpha channel.
  :param size: Output image size as (width, height)
  :type size: 2-tuple of int
  :param float ds: Object scaling in the interval (0, 1].
  :param int dx: Object translation (in pixels) in x-coordinate.
  :param int dy: Object translation (in pixels) in y-coordinate.
  :rtype: Image
  :returns: New image with mode 'RGBA'.

  """
  width, height = size
  assert 0 < ds and ds <= 1
  # Scale image
  size = np.array(img.size) * float(ds)
  img = img.resize(size.astype(int), Image.ANTIALIAS)  # down-sample image
  assert width >= img.size[0], "Object must fit within background"
  assert height >= img.size[1], "Object must fit within background"
  # Pad/crop image to fit background
  x_in_idx, x_out_idx = _ComputeIndices(img.size[0], width, dx)
  y_in_idx, y_out_idx = _ComputeIndices(img.size[1], height, dy)
  data_in = fromimage(img.convert('RGBA'))
  data_out = np.zeros(shape = (height, width, 4), dtype = data_in.dtype)
  # Copy image data into sub-range of output buffer
  data_out[y_out_idx, x_out_idx, :] = data_in[y_in_idx, x_in_idx, :]
  img_out = toimage(data_out)
  return img_out

def CombineImages(masked_object_img, bg_img, ds, dy, dx):
  """Create composite of object foreground and given background."""
  fg_img = _MakeObjectImage(masked_object_img, bg_img.size, ds, dy, dx)
  return Image.composite(fg_img, bg_img.convert('RGBA'), fg_img)

def MakeNoiseBg(size):
  """Create an image filled with 1/f^2 noise.

  :param size: Image size as (width, height).
  :type size: 2-tuple of int
  :rtype: Image
  :returns: Created image with mode 'L'.

  """
  from glimpse.util.gimage import MakeOneOverFNoise
  return toimage(MakeOneOverFNoise(size[::-1], -2))  # 1/f^2 noise

def ObjToStr(obj, *fields):
  cname = type(obj).__name__
  fields = ", ".join("%s = %s" % (f, getattr(obj, f)) for f in fields)
  return "%s(%s)" % (cname, fields)

class MaskedCorpus(object):

  def __init__(self, root, image_loader = None):
    assert os.path.isdir(root), "Can't find masked corpus directory: %s" % root
    if image_loader is None:
      image_loader = LoadImage
    self.root = root
    self.load_image = image_loader

  def __str__(self):
    return ObjToStr(self, 'root')

  def GetPath(self, object_id, rotation):
    """Construct a path to an image in the corpus directory."""
    return os.path.join(self.root, str(object_id),
        "%s_r%s.png" % (object_id, rotation))

  def Get(self, object_id, rotation):
    """Get an image from the corpus directory."""
    return self.load_image(self.GetPath(object_id, rotation))

  def GetRotations(self, object_id):
    """Get the set of available rotations in this directory.

    :param int object_id: Unique ID of ALOI object.
    :rtype: list of int
    :returns: Available rotations, in degrees.

    """

    def toint(x):
      try:
        return int(x)
      except:
        return None

    prefix = '%s_' % object_id
    postfix = ".png"
    pattern = os.path.join(self.root, str(object_id),
        '%sr*%s' % (prefix, postfix))
    paths = glob.glob(pattern)
    paths = map(os.path.basename, paths)
    prefix_len = len(prefix)
    postfix_len = len(postfix)
    rotations = [ toint(p[prefix_len+1 : -postfix_len]) for p in paths ]
    assert all((r is not None) for r in rotations)
    return rotations

  def GetObjectIds(self):
    """Get the set of available objects in this directory.

    :rtype: list of str
    :returns: List of found object IDs.

    """
    return [ p for p in os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root, p)) ]

class Renderer(object):
  """Render transformed objects over various backgrounds."""

  BG_BLACK = 'black'
  BG_NOISE = 'noise'

  # Dimensions of generated images.
  width = 200
  height = 200

  #: (str) Default background type.
  bg_type = BG_BLACK

  def __init__(self, corpus):
    self.corpus = corpus

  def __str__(self):
    return ObjToStr(self, 'corpus')

  __repr__ = __str__

  def MakeBackground(self, bg_type = None):
    """Create a background image.

    :param bg_type: The type of background to create: (BG_BLACK or BG_NOISE).
    :type bg_type: str

    """
    if bg_type == None:
      bg_type == self.bg_type
    if bg_type == self.BG_BLACK:
      return Image.new('L', size = (self.width, self.height), color = 0)
    elif bg_type == self.BG_NOISE:
      return MakeNoiseBg((self.width, self.height))
    elif isinstance(bg_type, basestring):
      return Image.open(bg_type).convert('L')
    raise ValueError("Unknown background type: %s" % bg_type)

  def Render(self, object_id, dr, ds, dy, dx, bg_image = None):
    """Render a single object on top of a given background image.

  :param int object_id: Unique identifier of ALOI object.
  :param int dr: Object rotation in degrees.
  :param float ds: Object scaling in the interval (0, 1].
  :param int dx: Object translation (in pixels) in x-coordinate.
  :param int dy: Object translation (in pixels) in y-coordinate.
    :param bg_image: Either an (Image) image to use as background (with size at
       least 768x576 -- the size of the ALOI object/mask images), or a
       background type for a newly-generated background image.

    See :method:`GetRotations` for list of available object rotations.

    """
    assert 0 < ds and ds <= 1
    bg = self.MakeBackground(bg_image)
    width, height = bg.size
    fg = self.corpus.Get(object_id, dr)
    img = CombineImages(fg, bg, ds, dy, dx)
    return img

  def RenderRandom(self, object_id, range_dr, range_ds, range_dy, range_dx,
      bg_images):
    """Render a set of synthesized ALOI images.

    Parameters with the name range_XX give the interval of values from which to
    sample, given in normalized units of variation. For range_dx and range_dy,
    the unit is one bounding box (width or height, respectively).

    """
    rotations = self.corpus.GetRotations(object_id)
    assert(len(rotations) > 0)
    rotations = [ r for r in rotations if r in range_dr ]
    assert(len(rotations) > 0)
    results = []
    for bg_image in bg_images:
      args = dict(
          object_id = object_id,
          dr = rotations[ np.random.randint(0, len(rotations)) ],
          ds = range_ds.Sample(),
          dx = int(range_dx.Sample()),
          dy = int(range_dy.Sample()),
          bg_image = bg_image,
      )
      img = self.Render(**args)
      results.append((args, img))
    return results

  def _RenderVarClass(self, object_id, vr, vs, vy, vx, bg_images):
    """A helper method to render a 'variation class' of object presentations.

    :param int object_id: Unique ALOI object ID.
    :param int vr: Rotational variation. Final variation will be drawn from the
       range [-vr, vr] mod 360.
    :param float vs: Scale variation. Final variation will be drawn from
       scale * ((1-vs), (1+vs)), where 'scale' is a constant factor that resizes
       the object's longest edge to 100px.
    :param int vy: Vertical translation. Final variation drawn from [-vy, vy).
    :param int vx: Horizontal translation. Final variation drawn from [-vx, vx).
    :param bg_images: Paths to background images, or the special values BG_BLACK
       or BG_NOISE.
    :type bg_images: list of str

    The object image will be scaled to have a long edge of 100px before
    requested scaling (vs) is applied.

    """
    fg = self.corpus.Get(object_id, 0)
    scale = 100. / max(fg.size)  # scale longest edge to 100px
    dr = ModRange(-vr, vr + 1, 360)
    ds = Range((1 - vs) * scale, (1 + vs + .001) * scale)
    dx = Range(-vx, vx)
    dy = Range(-vy, vy)
    return self.RenderRandom(object_id, dr, ds, dy, dx, bg_images)

  def RenderVar0(self, object_id, bg_images):
    """Render variation level 0.

      x-axis translation: 0px
      y-axis translation: 0px
      scaling: 0%
      rotation: 0 degrees

    """
    return self._RenderVarClass(object_id, 0, 0, 0, 0, bg_images)

  def RenderVar1(self, object_id, bg_images):
    """Render variation level 1.

      x-axis translation: +- 10 px
      y-axis translation: +- 10 px
      scaling: 10%
      rotation: 15 degrees

    """
    return self._RenderVarClass(object_id, 15, .1, 10, 10, bg_images)

  def RenderVar2(self, object_id, bg_images):
    """Render variation level 2.

      x-axis translation: +- 20 px
      y-axis translation: +- 20 px
      scaling: 20%
      rotation: 30 degrees

    """
    return self._RenderVarClass(object_id, 30, .2, 20, 20, bg_images)

  def RenderVar3(self, object_id, bg_images):
    """Render variation level 3.

      x-axis translation: +- 30 px
      y-axis translation: +- 30 px
      scaling: 30%
      rotation: 45 degrees

    """
    return self._RenderVarClass(object_id, 45, .3, 30, 30, bg_images)

  def RenderVar4(self, object_id, bg_images):
    """Render variation level 4.

      x-axis translation: +- 40 px
      y-axis translation: +- 40 px
      scaling: 40%
      rotation: 60 degrees

    """
    return self._RenderVarClass(object_id, 60, .4, 40, 40, bg_images)

  def RenderVar5(self, object_id, bg_images):
    """Render variation level 5.

      x-axis translation: +- 50 px
      y-axis translation: +- 50 px
      scaling: 50%
      rotation: 75 degrees

    """
    return self._RenderVarClass(object_id, 75, .5, 50, 50, bg_images)

  def RenderVar6(self, object_id, bg_images):
    """Render variation level 6.

      x-axis translation: +- 60 px
      y-axis translation: +- 60 px
      scaling: 60%
      rotation: 90 degrees

    """
    return self._RenderVarClass(object_id, 90, .6, 60, 60, bg_images)

def Render(base_dir, object_id, var_class, out_dir, *bg_images):
  """Render a set of images by pasting the object over background images.

  For each background image, the foreground object will be randomly transformed
  via (y-axis) rotation, scaling, and both vertical and horizontal translation.
  The degree of variation for each type of transformation depends on the
  variation class (var_class).

  :param str base_dir: Path to directory containing masked object images.
  :param object_id: Unique identifier of foreground object.
  :param int var_class: Level of variation (0-6) of generated images, with
     larger values indicating higher degree of variation.
  :param str out_dir: Path to output directory.
  :type bg_images: list of str
  :param bg_images: Path to background images. Can also be one of 'black' or
     'noise' for a flat-black background or a 1/f noise background,
     respectively.

  The base_dir directory should contain one level of sub-directories, whose
  names correspond to object IDs. Each sub-directory should contain files of the
  form "{object_id}_r{rotation}.png", with curly brackets for illustration.

  Results will be written to the output directory with file names of the form
  "{index}_{object_id}_r{rotation}s{scale}y{shifty}x{shiftx}_{bg_image}.png",
  where rotation, scale, shifty and shiftx indicate the degree of variation of
  the corresponding transformations.

  Example:

  Generate two images with black backgrounds from variation level three.

  >>> images = Render('/aloi/masked_objects', 139, 3, '/aloi/synthetic',
      ['black', 'black'])

  Generate one image with an existing background from variation level six.

  >>> image, = Render('/aloi/masked_objects', 139, 6, '/aloi/synthetic',
     ['/aloi/backgrounds/bg1.jpg'])

  """
  object_id = int(object_id)
  r = Renderer(MaskedCorpus(base_dir, LoadCachedImage))
  try:
    func = getattr(r, 'RenderVar%s' % var_class)
  except AttributeError:
    sys.exit("Unknown variation class: %s" % var_class)
  results = func(object_id, bg_images)
  idx = 1
  for args, img in results:
    args['fname'], _ = os.path.splitext(os.path.basename(args['bg_image']))
    args['idx'] = idx
    idx += 1
    path = "%(idx)s_%(object_id)s_r%(dr)03ds%(ds)1.3fy" % args
    path += "%(dy)03.fx%(dx)03.f_%(fname)s.png" % args
    path = os.path.join(out_dir, path)
    print os.path.basename(path)
    img.save(path)

if __name__ == '__main__':
  if len(sys.argv) < 6:
    sys.exit("usage: %s OBJ-ROOT OBJ-ID VAR-CLASS OUT-DIR BG-IMAGE ..." % \
        sys.argv[0])
  Render(*sys.argv[1:])

"""

ALOI contains rotations ranging over 360 degress, with 5 degree increments.
try with dr = dx = dy = 0 on 5 noise backgrounds. Note: we only have in-depth
rotation (right? or is this in-plane rotation?).

var0 -> no change in variation
var1 -> one "unit" of change, given as pixel-wise Euclidean distance between
        images with horizontal shift of one bounding box.

Euclidean distance for shifted objects:
[9164.9147558632249,
 495.97484044598201,
 3211.2170088426778,
 1530.6842291425673,
 214.65405613226022,
 1599.759323337156,
 7216.9126028447463,
 491.93347174164501,
 9298.2243137254245,
 4400.0957170318316,
 463.01730103806273,
 569.34013071896072,
 4655.0361860822049,
 2660.5240753556891,
 1756.1614763552293,
 2850.1864206073774,
 3386.3120030758773,
 6247.8547020376709]
The variation is quite high, so we should choose degree of change in an ad hoc
way.

Car v. Plane Variation
  x-axis: 0% - 60%
  y-axis: 0% - 120%
  scale:  0% - 60%
  rotation: 0 deg - 90 deg

Variation Level 0
  x-axis: +-0% of bounding box
  y-axis: +-0% of bounding box
  scale:  +-0%  (.7 - 1.3)
  rotation: +-0 deg

Variation Level 3
  x-axis: +-30% of bounding box
  y-axis: +-60% of bounding box
  scale:  +-30%  (.7 - 1.3)
  rotation: +-45 deg

Variation Level 4
  x-axis: +-40% of bounding box
  y-axis: +-80% of bounding box
  scale:  +-40%  (.7 - 1.3)
  rotation: +-60 deg

Variation Level 5
  x-axis: +-50% of bounding box
  y-axis: +-100% of bounding box
  scale:  +-50%  (.7 - 1.3)
  rotation: +-50 deg

Variation Level 6
  x-axis: +-60% of bounding box
  y-axis: +-120% of bounding box
  scale:  +-60%  (.7 - 1.3)
  rotation: +-90 deg

need to increase and decrease scale of object. the best way to do this is to
treat the original object scale as the largest scale (1.3x) of the target. thus,
the scales would range from (0.7/1.3 = 0.54x) to 1.0, where 0.7 is the smallest
scale.

dx is drawn from [-X, X], where X = 0.6 * bbox-width, and similarly for dy.
Per-object bounding box sizes, given as width x height
     9: 634x353
    62: 401x439
    93: 644x447
   113: 594x318
   136: 489x212
   138: 690x287
   139: 246x351
   145: 247x354
   151: 603x328
   154: 655x327
   158: 606x241
   160: 707x306
   354: 489x381
   390: 169x223
   482: 433x178
   539: 425x562
   543: 487x274
   546: 380x291
we can either use a per-object dx/dy, or choose a mean bbox size for all
objects.


For Pinto et al's Car v. Plane:
  side view is 100x50
For object #9, ds = 0.16 gives a bounding box of 101x56, which seems close
enough. If we want to vary this by +- 30%, then we'd use a range of
(.7*.16, 1.3*.16) = (.11, .21)


"""
