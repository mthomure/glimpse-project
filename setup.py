#!/usr/bin/env python

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# To rebuild the "base backend" from cython source, set the environment variable
# BUILD_BASE_BACKEND=1.


try:
    import setuptools
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    import setuptools

from setuptools import setup, find_packages
from setuptools.extension import Extension

import os

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

setup_args = dict()
if os.environ.get('BUILD_BASE_BACKEND') == "1":
  from Cython.Distutils import build_ext
  import numpy
  base_backend_module_file = "module.pyx"
  setup_args.update(cmdclass = {'build_ext': build_ext},
      include_dirs = [ numpy.get_include() ])
else:
  base_backend_module_file = "module.cpp"

base_backend_ext = NumpyExtension(
  "glimpse.backends._base_backend",
  [
    "glimpse/backends/base_backend/%s" % base_backend_module_file,
    "glimpse/backends/base_backend/array.cpp",
    "glimpse/backends/base_backend/bitset_array.cpp",
    "glimpse/backends/base_backend/filter.cpp",
    "glimpse/backends/base_backend/util.cpp",
  ],
  depends = [
    "glimpse/backends/base_backend/array.h",
    "glimpse/backends/base_backend/bitset_array.h",
    "glimpse/backends/base_backend/filter.h",
    "glimpse/backends/base_backend/util.h",
  ],
  language = "c++",
  extra_compile_args = extra_compile_args,
  extra_link_args = extra_link_args,
  define_macros = define_macros,
  undef_macros = undef_macros,
)

setup(
  name = "glimpse",
  version = "0.2.1",
  author = "Mick Thomure",
  author_email = "thomure@cs.pdx.edu",
  packages = find_packages(),
  url = 'https://github.com/mthomure/glimpse-project',
  license = 'LICENSE.txt',
  description = "Hierarchical visual models in C++ and Python",
  long_description = open('README.txt').read(),
  platforms = ["Linux", "Mac OS-X", "Unix"],
  entry_points = {
    'console_scripts' :
        ['glab = glimpse.glab.cli:Main']},
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
    "Python >= 2.7",
    "numpy",
    "scipy",
    "PIL",
    "traits",
    "scikit-learn",
    "decorator",
    "traitsui",
  ],
  setup_requires = [
    "numpy",
  ],
  package_data = {
    'glimpse.corpora' : ['data/*/*/*', 'data/*.txt'],
  },
  ext_modules = [ base_backend_ext ],
  **setup_args
)
