import re
import sys
import glimpse
import os

msg = sys.stdin.read()
pattern = re.compile(
  r'^.*?\(([^\+]+)\+(0x[^\)]+)\).*$',
  #~ r'^(/.{2}).*?\(([^\<\(]+).*\+(0x[^\)]+)\).*$',
  re.MULTILINE
)

import subprocess
def LookupOffsets(offsets):
  """Convert offset to name and line of source file, if possible."""
  p = subprocess.Popen(["addr2line", "-iCs", "-e", glimpse.__file__], stdout = subprocess.PIPE, stderr = subprocess.PIPE, stdin = subprocess.PIPE)
  (pout, perr) = p.communicate(offsets)
  assert (p.returncode == 0 or p.returncode == None), "Command exited with error:\n\t%s" % " ".join(perr)
  return re.sub(re.escape("??:0"), "", pout.strip())

def DemangleFunctions(lines):
  """Demangle C++ symbol names."""
  p = subprocess.Popen(["c++filt"], stdout = subprocess.PIPE, stderr = subprocess.PIPE, stdin = subprocess.PIPE)
  (pout, perr) = p.communicate(lines)
  assert (p.returncode == 0 or p.returncode == None), "Command exited with error:\n\t%s" % " ".join(perr)
  return pout

def ParseMessage(msg):
  lines = msg.split("\n")
  if len(lines) < 2 or lines[1] != 'Stack trace below:':
    return msg
  msg = "\n".join(lines[2:])
  hdr = "\n".join(lines[0:2]) + "\n"
  #~ hdr = ""
  pattern = r'^.*?%s\(([^\+]*)\+(0x[^\)]+)\).*$' % re.escape(os.path.basename(glimpse.__file__))
  pattern = re.compile(
    pattern,
    re.MULTILINE
  )
  functions, offsets = zip(*[ m.groups() for m in re.finditer(pattern, msg) ])
  functions = [ f if f else "(unknown function)" for f in DemangleFunctions("\n".join(functions)).split("\n") ]
  offsets = [ o if o else "(unknown location)" for o in LookupOffsets("\n".join(offsets)).split("\n") ]
  return hdr + "\n".join("%s at %s" % (f, o) for f, o in zip(functions, offsets))

print ParseMessage(msg)
