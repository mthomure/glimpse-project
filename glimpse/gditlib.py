from glimpse import util
from glimpse.executors import multicore_executor
from glimpse.models import viz2
from glimpse import backends
import os
import svmutil

class ModelWrapper(object):
  """Simplified interface for a glimpse model."""

  def __init__(self, model = None):
    if model == None:
      # Create a default glimpse model.
      model = viz2.Model(backends.CythonBackend(), viz2.Params())
    self.model = model
    self.mapper = multicore_executor.MulticoreMap()
    self.model_transform = viz2.ModelTransform(model, viz2.Layer.C1,
        save_all = False)  # keep only the C1 layer data

  def ComputeC1Activity(self, input_states):
    """Compute the C1 layer activity for a set of images."""
    # Map the model transform (i.e., computing C1 data) over the states.
    output_states = self.mapper.Map(self.model_transform, input_states)
    # given output state s:
    #  c1 data is in a = s[ Layer.C1.id ].activity
    #  activity vector is given by util.ArrayListToVector(a)
    # convert each output state to a 1-dimensional vector of C1 activities
    return [ util.ArrayListToVector(s[ viz2.Layer.C1.id ].activity)
        for s in output_states ]

  def ImageToState(self, image):
    return self.model.MakeStateFromImage(image)

  def FilenameToState(self, filename):
    # Create an empty model state object for each image.
    return self.model.MakeStateFromFilename(filename)

class SvmFeatureTransformer(object):

  def ComputeSvmFeatures(self, c1_activity):
    """Compute the SVM feature vector for a single image."""
    return c1_activity

def BuildSvmFeatureTransformer(c1_activity_list):
  """Construct a new SVM feature transformer."""
  return SvmFeatureTransformer()

def TrainSvm(pos_features, neg_features):
  classes = [1] * len(pos_features) + [-1] * len(neg_features)
  features = pos_features + neg_features
  features = map(list, features)
  options = '-q -b 1'  # don't write to stdout
  return svmutil.svm_train(classes, features, options)

def TestSvm(model, pos_features, neg_features):
  classes = [1] * len(pos_features) + [-1] * len(neg_features)
  features = pos_features + neg_features
  features = map(list, features)
  # Sadly, we can't make it shut up -- i.e., it writes to stdout.
  options = '-b 1'
  predicted_labels, acc, decision_values = svmutil.svm_predict(classes,
      features, model, options)
  decision_values = [ dv[0] for dv in decision_values ]
  return predicted_labels, decision_values

def GetDirContents(dir_path):
  return [ os.path.join(dir_path, f) for f in os.listdir(dir_path) ]
