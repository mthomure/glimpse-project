#!/usr/bin/env python

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages
from setuptools.extension import Extension

#### Control Flags for Compilation ####

# Optimize runtime binary. Disables output needed for debugging. Implies that
# ndebug = False.
optimize = True

# Disable the NDEBUG preprocessor define, adding (expensive) sanity checks to
# resulting binary.
debug = (not optimize)

# Enable SSE intrinsics.
sse = True

# Add output required for profiling. This generally does not affect the speed of
# the resulting binary.
profiler = False


#### Parsing of Control Flags ####

extra_compile_args = []
extra_link_args = []
define_macros = []
undef_macros = []

if optimize:
  extra_compile_args += [ "-O3" ]
  extra_link_args += [ ]
else:
  extra_compile_args += ["-O0", "-g" ]
  extra_link_args += [ "-rdynamic", "-fno-omit-frame-pointer", "-g" ]

if debug:
  undef_macros += [ "NDEBUG" ]
else:
  define_macros += [ ("NDEBUG", None) ]

if sse:
  extra_compile_args += [ "-msse3", "-mfpmath=sse" ]
  extra_link_args += [ "-msse3" ]

if profiler:
  extra_compile_args += [ "-pg" ]
  extra_link_args += [ "-lprofiler", "-pg" ]


#### Do Not Edit Below This Line ####

# See http://permalink.gmane.org/gmane.comp.python.distutils.devel/5345
class NumpyExtension(Extension):
   def __init__(self, *args, **kwargs):
     Extension.__init__(self, *args, **kwargs)
     self._include_dirs = self.include_dirs
     del self.include_dirs # restore overwritten property

   # warning: Extension is a classic class so it's not really read-only
   @property
   def include_dirs(self):
     from numpy import get_include
     return self._include_dirs + [get_include()]

cython_backend_ext = NumpyExtension(
  "glimpse.backends.cython_backend.filter",
  [
    "glimpse/backends/cython_backend/filter.cpp",
    "glimpse/backends/cython_backend/src/array.cpp",
    "glimpse/backends/cython_backend/src/bitset_array.cpp",
    "glimpse/backends/cython_backend/src/filter.cpp",
    "glimpse/backends/cython_backend/src/util.cpp",
  ],
  depends = [
    "glimpse/backends/cython_backend/src/array.h",
    "glimpse/backends/cython_backend/src/bitset_array.h",
    "glimpse/backends/cython_backend/src/filter.h",
    "glimpse/backends/cython_backend/src/util.h",
  ],
  language = "c++",
  extra_compile_args = extra_compile_args,
  extra_link_args = extra_link_args,
  define_macros = define_macros,
  undef_macros = undef_macros,
)

setup(
  name = "glimpse",
  version = "0.1.0",
  author = "Mick Thomure",
  author_email = "thomure@cs.pdx.edu",
  packages = find_packages(),
  url = 'https://github.com/mthomure/glimpse-project',
  license = 'LICENSE.txt',
  description = "Library for hierarchical visual models in C++ and Python",
  long_description = open('README.txt').read(),
  platforms = ["Linux", "Mac OS-X", "Unix"],
  classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: C++",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
  ],
  install_requires = [
    "Python >= 2.6",
    "numpy >= 1.3",
    "scipy >= 0.7.2",
    "PIL >= 1.1.6",
    "traits >= 3.4",
  ],
  setup_requires = [
    "numpy >= 1.3",
  ],
  extras_require = {
    "gui" : [
      "traitsui >= 3.4",
    ],
    "cluster" : [
      "gearman >= 0.14",
      "pyzmq >= 2.1.11",
    ],
  },
  entry_points = {
    'console_scripts' : [
      'glab = glimpse.glab:main',
      'glimpse-cluster = glimpse.pools.main:main [cluster]',
    ],
    'gui_scripts' : [
      'edit-glimpse-params = glimpse.models.edit_params:main [gui]',
    ],
  },
  ext_modules = [ cython_backend_ext ],
)
