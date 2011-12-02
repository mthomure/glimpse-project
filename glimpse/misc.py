# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import os

def GetExampleImagePaths():
  """Get the path of example images in the Glimpse project."""
  # The root of the Glimpse project
  root = os.path.dirname(os.path.dirname(__file__))
  # The RC directory within the Glimpse project
  rc = os.path.join(root, "rc")
  return [ os.path.join(rc, f) for f in ("example.jpg", "example.tif") ]
