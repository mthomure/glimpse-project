from contextlib import contextmanager
import importlib
import os
import unittest

from .misc import *
import glimpse.pools.gearmancluster

@contextmanager
def OverrideVar(name, value):
  orig_value = globals()[name]
  globals()[name] = value
  yield
  globals()[name] = orig_value

class TestCase(unittest.TestCase):

  def testDefaults(self):
    # verify assumption that we use a multicore pool by default
    self.assertEqual(DEFAULT_POOL_TYPE, "multicore")
    self.assertEqual(DEFAULT_CLUSTER_TYPE, "ipython")

  def testMakePool_default(self):
    if 'GLIMPSE_POOL' in os.environ:
      del os.environ['GLIMPSE_POOL']
    self.assertIsInstance(MakePool(), MulticorePool)

  def testMakePool_withArg(self):
    if 'GLIMPSE_POOL' in os.environ:
      del os.environ['GLIMPSE_POOL']
    self.assertIsInstance(MakePool("singlecore"), SinglecorePool)

  def testMakePool_withEnvVar(self):
    os.environ['GLIMPSE_POOL'] = "singlecore"
    self.assertIsInstance(MakePool(), SinglecorePool)

  def testGetClusterPackage_default(self):
    if 'GLIMPSE_CLUSTER' in os.environ:
      del os.environ['GLIMPSE_CLUSTER']
    self.assertEqual(GetClusterPackage(), glimpse.pools.ipythoncluster)

  def testGetClusterPackage_withArg(self):
    if 'GLIMPSE_CLUSTER' in os.environ:
      del os.environ['GLIMPSE_CLUSTER']
    with OverrideVar('DEFAULT_CLUSTER_TYPE', ""):
      self.assertEqual(GetClusterPackage("gearman"),
          glimpse.pools.gearmancluster)

  def testGetClusterPackage_withEnvVar(self):
    os.environ['GLIMPSE_CLUSTER'] = "gearman"
    with OverrideVar('DEFAULT_CLUSTER_TYPE', ""):
      self.assertEqual(GetClusterPackage(), glimpse.pools.gearmancluster)

if __name__ == '__main__':
  unittest.main()
