# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.models.viz2.model import Layer
from glimpse.models.viz2.model import Model as BaseModel
from glimpse.models.viz2.model import State as BaseState
from ops import ModelOps
from params import Params

class State(BaseState):
  """A container for the :class:`Model` state."""
  pass

class Model(ModelOps, BaseModel):
  """Create a 2-part, HMAX-like hierarchy of S+C layers.

  This module implements the configuration used by Mutch & Lowe (2008).

  """

  #: The datatype associated with network states for this model.
  StateClass = State

# Add (circular) Model reference to State class.
State.ModelClass = Model
