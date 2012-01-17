#!/usr/bin/env python

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

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
  extra_compile_args += [ "-O3", "-mtune=native" ]
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

cython_backend_ext = Extension(
  "glimpse.backends.cython_backend.filter",
  [
    "glimpse/backends/cython_backend/filter.pyx",
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
  version = "1.1",
  description = "Library for hierarchical visual models in C++ and Python",
  author = "Mick Thomure",
  author_email = "thomure@cs.pdx.edu",
  cmdclass = {'build_ext': build_ext},

  ext_modules = [ cython_backend_ext ],
  packages = [ 'glimpse', 'glimpse.backends', 'glimpse.backends.cython_backend',
      'glimpse.models', 'glimpse.models.viz2', 'glimpse.models.ml',
      'glimpse.pools', 'glimpse.pools.zmq_cluster', 'glimpse.pools.gearman_cluster',
      'glimpse.util' ],
  include_dirs = [ numpy.get_include() ],
)
