# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# Create a 2-part, HMAX-like hierarchy of S+C layers.
# This module implements the configuration used by Mutch & Lowe (2008).

from glimpse.models.viz2.model import Layer
from glimpse.models.viz2.model import Model as BaseModel
from glimpse.models.viz2.model import State as BaseState
from ops import ModelOps
from params import Params

class State(BaseState):
  """A container for the model state. Each model has a seperate State object, so
  it is always clear which model generated a given state object."""
  pass

class Model(ModelOps, BaseModel):

  # The datatype associated with network states for this model.
  State = State

# Add (circular) Model reference to State class.
State.Model = Model
