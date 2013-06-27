"""This module fixes a strange bug when importing PIL.

This is a workaround for a strange bug seen on Mac OS X, in which the
PIL._imaging module is initialized more than once when PIL is imported in
different ways (e.g., Image vs PIL.Image). This bug results in the fatal error
  AccessInit: hash collision: 3 for both 1 and 1
XXX For some reason, this prevents the error on "import PIL.Image", but not on
"from PIL import Image".

"""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import Image
import PIL
PIL.Image = Image
