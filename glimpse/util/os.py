
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Functions for operating system and filesystem interactions.
#

import os
import shutil
import subprocess
import sys
import tempfile

class TempDir(object):
  """A temporary directory, which is deleted when this object is destroyed. For
  a temporary file, see the tempfile.TemporaryFile class."""
  def __init__(self):
    self.dir = tempfile.mkdtemp()
  def __del__(self):
    shutil.rmtree(self.dir)
  def MakePath(self, relPath):
    return os.path.join(self.dir, relPath)
  def CopyFrom(self, srcPath, destRelPath):
    '''Copy a single file into the temporary directory.'''
    shutil.copyfile(srcPath, self.MakePath(destRelPath))
  def CopyAllTo(self, destPath):
    '''Copy all files in temporary directory to another location.'''
    RunCommand("cp %s/* %s/" % (self.dir, destPath))
    pass
  def Remove(self, relPath):
    '''Remove a single file from the temporary directory.'''
    os.remove(self.MakePath(relPath))
  def GetDir(self):
    return self.dir

def RunCommand(cmd, verbose=False, catch_retval=True):
  """Execute a command on the shell, and check the results. See the Cmd class
  for a more flexibile solution."""
  if verbose:
    print >>sys.stderr, "CMD: " + cmd
  retval = os.system(cmd)
  if catch_retval and retval != 0:
    raise Exception("Command returned non-zero status:\nCMD: %s\nRETURN: %s" % (cmd, retval))
  return retval == 0

class Cmd(object):
  """A builder syntax for shell commands.

    Bash Shell Syntax         Python Builder Syntax
    ---------------------------------------------------------------------
    C                         Cmd("C").Run()
    C | P                     Cmd("C").ReturnStdout()
    C 1>F1                    Cmd("C").Redirect("F")
    C 1>F1 2>F2               Cmd("C").Redirect("F1", "F2")
    C1 | C2 1>F               Cmd("C2").PipeFrom(Cmd("C1")).Redirect("F")
    C 1>F 2>&1                Cmd("C").TieStderrToStdout().Redirect("F")
    C 2>&1 | P                Cmd("C").TieStderrToStdout().ReturnStdout()
    C 2>&1 1>/dev/null | P    Cmd("C").TieStderrToStdout().ReturnStderr()"""

  def __init__(self, *args):
    """Construct a command object.
    args -- (list) arguments to subprocess.Popen(), with path to binary as first
            value
    """
    self.args = args
    self.cmd_str = " ".join(args)
    self.popen_stdout = None
    self.popen_stderr = None
    self.communicate_stdout = None
    self.communicate_stderr = None
    self.verbose = False

  def TieStderrToStdout(self):
    self.popen_stderr = subprocess.STDOUT

  def Run(self):
    if self.verbose:
      print "CMD: %s" % self.cmd_str
    p = subprocess.Popen(self.args, stdout = self.popen_stdout, stderr = self.popen_stderr)
    (self.communicate_stdout, self.communicate_stderr) = p.communicate()
    assert(p.returncode == 0 or p.returncode == None), "Command exited with error:\n\t%s" % " ".join(self.args)

  def ReturnStdout(self):
    """Run command, capturing and returning stdout."""
    self.popen_stdout = subprocess.PIPE
    self.Run()
    return self.communicate_stdout

  def ReturnStderr(self):
    """Run command, capturing and returning stdout."""
    self.popen_stderr = subprocess.PIPE
    self.Run()
    return self.communicate_stderr

  def Redirect(self, fout = None, ferr = None, fout_append = None, ferr_append = None):
    """Run command, redirecting stdout and/or stderr to a file."""
    redir_strs = []
    if fout:
      redir_strs.append("1>%s" % fout)
      self.popen_stdout = open(fout, 'w')
    elif fout_append:
      redir_strs.append("1>>%s" % fout_append)
      self.popen_stdout = open(fout_append, 'w+')
    if ferr:
      redir_strs.append("2>%s" % ferr)
      self.popen_stderr = open(ferr, 'w')
    elif ferr_append:
      redir_strs.append("2>>%s" % ferr_append)
      self.popen_stderr = open(ferr_append, 'w+')
    self.cmd_str = " ".join(list(self.args) + redir_strs)
    self.Run()
    if self.popen_stdout:
      self.popen_stdout.close()
      self.popen_stdout = None
    if self.popen_stderr:
      self.popen_stderr.close()
      self.popen_stderr = None

def IsImageFile(fname):
  """Determine if a file contains an image, based on filename's extension."""
  image_file_extensions = [ ".jpg", ".jpeg", ".png", ".tif", ".tiff" ]
  return any( map(fname.lower().endswith, image_file_extensions) )

def TouchFile(fname):
  """Update a file's access time, creating the file if it does not exist."""
  fh = file(fname, 'a')  # create file, if it doesn't exist
  try:
    os.utime(fname, None)
  finally:
    fh.close()
