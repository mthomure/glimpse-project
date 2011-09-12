
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Classes and functions for I/O and evaluation of linear support vector machines
# (SVMs).
#

from glimpse import util
import numpy
import sys

class SvmModel(object):
  """Represents a two-class linear kernel SVM."""
  def __init__(self, coefficients, support_vectors, bias):
    self.coefficients = coefficients
    self.support_vectors = support_vectors
    self.bias = bias
  def Classify(self, xs):
    return Sign(self.support_vectors.dot(xs).dot(self.coefficients) - self.bias)
  def Evaluate(self, xs):
    return numpy.dot(
        numpy.dot(self.support_vectors, xs),
        self.coefficients) - self.bias

class SvmScaler(object):
  def __init__(self, fname):
    lines = util.ReadLines(fname)
    self.out_low, self.out_high = map(float, lines[1].split())
    self.in_ranges = [ map(float, line.split()[1:]) for line in lines[2:] ]
  def _Scale(self, idx, value):
    low, high = self.in_ranges[idx]
    if value < low:
      value = low
    elif value > high:
      value = high
    normalized_input = (value - low) / (high - low)
    normalized_output = normalized_input * (self.out_high - self.out_low)
    return normalized_output + self.out_low
  def Scale(self, vector):
    scaled_vector = [ self._Scale(i,x) for i,x in zip(range(len(vector)), vector) ]
    return numpy.array(scaled_vector, dtype=numpy.float32)

def Sign(x):
  if x < 0:
    return -1
  elif x > 0:
    return 1
  return 0

def LoadSvmInstances(fname = sys.stdin):
  """Reads SVM instance lines from file (name or handle). Lines must be
  formatted as
    VAL KEY:VAL [...] [# ...]
  where VAL is a float, KEY is an int, and everything after the "#" is ignored.
  If lines are support vectors, then first (header) VAL gives the alpha
  coefficient. Otherwise, lines are training or testing instances and first VAL
  is target class.
  @returns list of header values and another list of feature vectors"""

  def ParseLine(line):
    line = line.strip().split()
    assert(len(line) > 1)
    header = float(line[0])
    keys = []
    values = []
    for entry in line[1:]:
      if entry == "#":
        break
      key, value = entry.split(":")
      keys.append(int(key) - 1)   # LIBSVM indices are 1-based
      values.append(float(value))
    features = numpy.empty([max(keys)+1], numpy.float32)
    features[keys] = values
    return header, features

  if hasattr(fname, 'read'):
    fh = fname
  else:
    fh = open(fname, 'r')
  values = map(ParseLine, fh)
  headers = numpy.array(list(v[0] for v in values), numpy.float32)
  features = numpy.array(list(v[1] for v in values), numpy.float32)
  if fh != fname:
    fh.close()
  return headers, features

LoadInstances = LoadSvmInstances

def StoreSvmInstances(headers, features, fname = sys.stdout):
  """Serialize SVM instance data in LIBSVM/SVM-LIGHT format.
  @params headers per-instance target class (either +1 or -1)
  @params features per-instance feature vector
  @params fname output file name or handle"""
  if hasattr(fname, "write"):
    fh = fname
  else:
    fh = open(fname, "w")
  for h, f in zip(headers, features):
    # SVM keys are 1-based
    print >>fh, "%s %s" % (h,
      " ".join("%s:%s" % (k+1, v) for k, v in zip(range(len(f)), f)))
  if fh != fname:
    fh.close()

StoreInstances = StoreSvmInstances

def StoreSvmInstances2(pos_activity, neg_activity, fname):
  """Serialize SVM instance data in LIBSVM/SVM-LIGHT format.
  @params pos_activity feature vectors for positive instances
  @params neg_activity feature vectors for positive instances
  @params fname output file name or handle"""
  num_pos, num_neg = pos_activity.shape[0], neg_activity.shape[0]
  headers = [1] * num_pos + [-1] * num_neg
  num_features = pos_activity.shape[1]
  assert(num_features == neg_activity.shape[1])
  features = numpy.empty((num_pos + num_neg, num_features), numpy.float32)
  features[:num_pos] = pos_activity
  features[num_pos:] = neg_activity
  StoreSvmInstances(headers, features, fname)

def LoadLibsvmModel(fname = sys.stdin):
   fh = open(fname)
   # start read header
   if not (
     fh.readline().strip() == "svm_type c_svc" and
     fh.readline().strip() == "kernel_type linear" and
     fh.readline().strip() == "nr_class 2"):
     raise Exception("Bad LIBSVM header")
   line = fh.readline().strip().split()
   if not (len(line) == 2 and line[0] == "total_sv"):
     raise Exception("Bad LIBSVM header")
   num_vectors = int(line[1])
   line = fh.readline().strip().split()
   if not (len(line) == 2 and line[0] == "rho"):
     raise Exception("Bad LIBSVM header")
   bias = float(line[1])
   if not fh.readline().strip() == "label 1 -1":
     raise Exception("Bad LIBSVM header")
   line = fh.readline().strip().split()
   if not (len(line) == 3 and line[0] == "nr_sv" and \
       int(line[1]) + int(line[2]) == num_vectors):
     raise Exception("Bad LIBSVM header")
   if not fh.readline().strip() == "SV":
     raise Exception("Bad LIBSVM header")
   # end read header
   if num_vectors <= 0: # or math.abs(bias) > 0.0001:
     raise Exception("Bad LIBSVM header")
   coefficients, support_vectors = LoadSvmInstances(fh)
   fh.close()
   return SvmModel(coefficients, support_vectors, bias)

ReadLibsvmModel = LoadLibsvmModel

def StoreLibsvmModel(model, fname = sys.stdout):
  """Serialize SVM model to LIBSVM format."""
  if hasattr(fname, "write"):
    fh = fname
  else:
    fh = open(fname, "w")
  print >>fh, "svm_type c_svc"
  print >>fh, "kernel_type linear"
  print >>fh, "nr_class 2"
  print >>fh, "total_sv %d" % len(model.support_vectors)
  print >>fh, "rho %f" % model.bias
  print >>fh, "label 1 -1"

  raise Exception("Not implemented: need to find documentation on LIBSVM model formatting.")

  print >>fh, "nr_sv %d %d" % (num_pos, num_neg)
  print >>fh, "SV"
  StoreSvmInstances(model.coefficients, model.support_vectors, fh)
  if fh != fname:
    fh.close()

def StoreSvmlightModel(model, fname = sys.stdout):
  """Serialize SVM model to SVM-light format."""
  if hasattr(fname, "write"):
    fh = fname
  else:
    fh = open(fname, "w")
  print >>fh, "SVM-light Version V6.02"
  print >>fh, "0 # kernel type"
  print >>fh, "3 # kernel parameter -d"
  print >>fh, "1 # kernel parameter -g"
  print >>fh, "1 # kernel parameter -s"
  print >>fh, "1 # kernel parameter -r"
  print >>fh, "empty# kernel parameter -u"
  print >>fh, "%d # highest feature index" % model.support_vectors.shape[1]
  print >>fh, "%d # number of training documents" % 0
  print >>fh, "%d # number of support vectors plus 1" % (model.support_vectors.shape[0] + 1)
  print >>fh, "%f # threshold b, each following line is a SV (starting with alpha*y)" % model.bias
  StoreSvmInstances(model.coefficients, model.support_vectors, fh)
  if fh != fname:
    fh.close()

def LoadSvmlightModel(fname = sys.stdin):
  fh = open(fname)
  # start read header
  if not ((fh.readline().startswith("SVM-light Version") and
           fh.readline().strip().split()[0] == "0")):
    raise Exception("Bad SVM-light header")
  for x in range(5): fh.readline()
  num_features = int(fh.readline().strip().split(" ")[0])
  fh.readline()
  num_vectors = int(fh.readline().strip().split(" ")[0]) - 1
  bias = float(fh.readline().strip().split(" ")[0])
  # end read header
  coefficients, support_vectors = LoadSvmInstances(fh)
  fh.close()
  assert(num_features == support_vectors.shape[1]), "SVM-light format error"
  return SvmModel(coefficients, support_vectors, bias)

ReadSvmlightModel = LoadSvmlightModel

def ReadSvmFeatureString(feature_string, feature_array):
  line = feature_string.split()
  for entry in line:
    entry = entry.split(':')
    idx = int(entry[0]) - 1     # LIBSVM indices are 1-based
    value = float(entry[1])
    feature_array[idx] = value
  return feature_array

def FormatSvm(label, features):
  """Convert a single instance to a string in LIBSVM format.
  label -- either 1 or -1
  features -- array of feature values.
  """
  features = " ".join("%d:%s" % (i, f) for i,f in zip(range(1, len(features)+1),
      features))
  return "%s %s" % (label, features)

def CorpusToSvm(corpus_dir, fh):
  """Read IT activity for each image, iterating over directories in corpus_dir/{pos,neg}/*."""
  join = os.path.join
  def f(dir): return [ util.Unpickle(join(d, 'it-activity')) for d in glob(join(dir, '*')) ]
  for v in f(join(corpus_dir, 'pos')):
    print >>fh, FormatSvm(1, v)
  for v in f(join(corpus_dir, 'neg')):
    print >>fh, FormatSvm(-1, v)

#~ def ActivityToSvm(pos_activity, neg_activity, fname):
  #~ """Write a set of positive and negative instances to a file in LIBSVM format.
  #~ """
  #~ fh = open(fname, 'w')
  #~ for v in pos_activity:
    #~ print >>fh, FormatSvm(1, v)
  #~ for v in neg_activity:
    #~ print >>fh, FormatSvm(-1, v)
  #~ fh.close()

def ActivityToSvm(pos_activity, neg_activity, fname):
  num_pos, num_neg = len(pos_activity), len(neg_activity)
  headers = [1] * num_pos + [-1] * num_neg
  num_features = pos_activity[0].shape[0]
  assert(num_features == neg_activity[0].shape[0])
  features = numpy.empty((num_pos + num_neg, num_features), numpy.float32)
  features[:num_pos] = pos_activity
  features[num_pos:] = neg_activity
  StoreSvmInstances(headers, features, fname)

