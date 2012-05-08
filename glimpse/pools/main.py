"""Entry point for glimpse-cluster."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from . import misc

def main():
  pkg = misc.GetClusterPackage()
  pkg.RunMain()
