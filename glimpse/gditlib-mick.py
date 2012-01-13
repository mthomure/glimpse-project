from glimpse import util
from glimpse.util import svm
import multiprocessing
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
    self.pool = multiprocessing.Pool()
    self.model_transform = viz2.ModelTransform(model, viz2.Layer.C1,
        save_all = False)  # keep only the C1 layer data

  def ComputeC1ActivityIterable(self, input_states):
    """Compute the C1 layer activity for a set of images."""
    output_states = self.pool.imap(self.model_transform, input_states)
    return ( util.ArrayListToVector(s[ viz2.Layer.C1.id ].activity)
        for s in output_states )

  def ComputeC1ActivityList(self, input_states):
    """Compute the C1 layer activity for a set of images."""
    # Map the model transform (i.e., computing C1 data) over the states.
    output_states = self.pool.map(self.model_transform, input_states)
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

#~ class SvmFeatureTransformer(object):
#~
  #~ def ComputeSvmFeatures(self, c1_activity):
    #~ """Compute the SVM feature vector for a single image."""
    #~ return c1_activity

def BuildSvmFeatureTransformer(c1_activity_list):
  """Construct a new SVM feature transformer."""
  scaler = svm.Sphering
  return SvmFeatureTransformer()

# XXX Enabling this flag (i.e., passing "-b 1" to LIBSVM) seems to cause a
# significant change in SVM performance. The reason for this is unclear.
COMPUTE_CONFIDENCE = False

# performs training of the svm classifier with scaling of the features
def TrainSvm(pos_features, neg_features):
  model = svm.ScaledSvm()
  model.Train((pos_features, neg_features))
  return model

def TestSvm(model, pos_features, neg_features):
  return model.Test((pos_features, neg_features))

# apply the model to the image
def ApplyModel(model,img_features):
  # features -- list of feature vectors, so len(features) is number of instances
  classes = [-1] * len(img_features)  # fake target class for svm_predict()
  features = map(list,img_features)
  options = ''
  predicted_labels, acc, decision_values = svmutil.svm_predict(classes,features,model,options)
  decision_values = [ dv[0] for dv in decision_values ]
  return predicted_labels, decision_values

def GetDirContents(dir_path):
  return [ os.path.join(dir_path, f) for f in os.listdir(dir_path) ]
