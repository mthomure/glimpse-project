import importlib
import os
import unittest

from .misc import *
from . import base

derived_name = DEFAULT_MODEL_NAME
derived = importlib.import_module(".%s" % derived_name, __package__)

class TestCase(unittest.TestCase):

  def testDefaultModelName(self):
    # verify assumption that we use a derived model by default
    self.assertNotEqual(DEFAULT_MODEL_NAME, "base")

  def testGetModelClass_default(self):
    if 'GLIMPSE_MODEL' in os.environ:
      del os.environ['GLIMPSE_MODEL']
    self.assertEqual(GetModelClass(), derived.Model)

  def testGetModelClass_withEnvVar(self):
    os.environ['GLIMPSE_MODEL'] = "base"
    self.assertEqual(GetModelClass(), base.Model)

  def testGetModelClass_withArg(self):
    os.environ['GLIMPSE_MODEL'] = derived_name
    self.assertEqual(GetModelClass("base"), base.Model)

  def testMakeParams_default(self):
    if 'GLIMPSE_MODEL' in os.environ:
      del os.environ['GLIMPSE_MODEL']
    self.assertIsInstance(MakeParams(), derived.Params)

  def testMakeParams_withArg(self):
    os.environ['GLIMPSE_MODEL'] = derived_name
    self.assertIsInstance(MakeParams("base"), base.Params)

  def testMakeModel_default(self):
    if 'GLIMPSE_MODEL' in os.environ:
      del os.environ['GLIMPSE_MODEL']
    self.assertIsInstance(MakeModel(), derived.Model)

  def testMakeModel_withArg(self):
    os.environ['GLIMPSE_MODEL'] = derived_name
    self.assertIsInstance(MakeModel("base"), base.Model)

  def testMakeModel_withArgAndParams(self):
    os.environ['GLIMPSE_MODEL'] = derived_name
    p = base.Params()
    m = MakeModel("base", p)
    self.assertIsInstance(m, base.Model)
    self.assertEqual(m.params, p)

  def testMakeModel_withParams(self):
    if 'GLIMPSE_MODEL' in os.environ:
      del os.environ['GLIMPSE_MODEL']
    p = derived.Params()
    m = MakeModel(p)
    self.assertIsInstance(m, derived.Model)
    self.assertEqual(m.params, p)

if __name__ == '__main__':
  unittest.main()
