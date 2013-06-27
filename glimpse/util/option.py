# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import getopt
import os
from pprint import pformat
import sys
import textwrap

# Maybe split option spec from option value?

__all__ = (
    'Option',
    'OptionRoot',
    'OptionGroup',
    'ImmutableAttributeSet',
    'OptionError',
    'OptValue',
    'BadOptionValue',
    'MissingOption',
    'PrintUsage',
    'ParseCommandLine',
    'UsageException',
    'Usage',
    'GetOptions',
    )

def FormatStr(obj):
  return type(obj).__name__

def FormatRepr(obj, args = None, kw = None, subargs = None,
    include_none = True):
  values = []
  if args is not None:
    values += [ pformat(getattr(obj, k)) for k in args ]
  if subargs is not None:  # format sequence attribute as arguments to init
    for k in subargs:
      vs = getattr(obj, k)
      if vs is not None:
        values += map(pformat, vs)
  if kw is not None:
    for k in kw:
      v = getattr(obj, k)
      if include_none or v is not None:
        values.append("%s=%s" % (k, pformat(v)))
  return "%s(%s)" % (type(obj).__name__, ", ".join(values))

class ImmutableAttributeSet(object):

  def __init__(self, kw1 = None, **kw2):
    """

    :type kw1: dict or sequence of 2-tuples

    """
    if kw1 is not None:
      self.__dict__.update(kw1)
    self.__dict__.update(kw2)

  def __delattr__(self, k):
    raise Exception("Attributes can not be deleted")

  def __setattr__(self, k, v):
    if k not in self.__dict__:
      raise AttributeError(k)  # disallow new attributes
    self.__dict__[k] = v

  def __repr__(self):
    return "%s(%s)" % (type(self).__name__, ", ".join("%s=%s" % (k, pformat(v))
        for k, v in self.__dict__.items()))

class OptionGroup(object):
  """A collection of options."""

  def __init__(self, name, *children):
    self.__dict__.update(_name = name, _children = children)
    for opt in children:
      self.__dict__[OptName(opt)] = opt

  def __iter__(self):
    return iter(self._children)

  def __str__(self):
    return FormatStr(self)

  def __repr__(self):
    return FormatRepr(self, ['_name'], subargs = ['_children'])

  def __setattr__(self, k, v):
    raise Exception("Option groups are immutable (though the options they "
        "contain are not) -- did you mean %s.value?" % k)

  @property
  def _value(self):
    return ImmutableAttributeSet(((opt._name, opt._value)
        for opt in self._children))

# test option group throws exception on setattr

def OptionRoot(*children):
  """Create the root option group."""
  return OptionGroup(None, *children)

def OptName(opt):
  return opt._name

def OptValue(opt):
  return opt._value

class Option(object):
  """A selection for some user-configurable program behavior."""

  def __init__(self, name, default = None, display = None, flag = None,
      doc = None, parser = 'type', multiple = False, enum = None):
    if flag is not None and len(flag) < 1:
      raise ValueError("Flag must be non-empty string")
    self._name = name  # preceeding underscore to match OptionGroup
    self.default = default
    self.display = display
    if (flag is not None and (not hasattr(flag, '__len__') or
        isinstance(flag, basestring))):
      flag = [flag]
    self.flag = flag
    flag_list = flag
    if not hasattr(flag_list, '__len__'):
      flag_list = [flag_list]
    for f in flag_list:
      if f.startswith('-'):
        raise ValueError("Flag should not start with hyphen")
    self.doc = doc
    if enum:
      edoc = "(one of: %s)" % ", ".join(map(str, enum))
      if self.doc:
        self.doc += " " + edoc
      else:
        self.doc = edoc
    self.parser = parser
    self.multiple = multiple
    self.enum = enum
    self._value = default  # preceeding underscore to match OptionGroup

  def __repr__(self):
    return FormatRepr(self, ('_name',), ('default', 'display', 'flag', 'doc',
        'parser', 'multiple', 'enum', '_value'), include_none = False)

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value_):
    value_ = self.FormatValue(value_)
    if self.multiple:
      if self._value is None:
        self._value = [value_]
      else:
        self._value.append(value_)
    else:
      self._value = value_

  def FormatValue(self, value):
    """Called when the option is detected in the command-line flags."""
    p = self.parser
    if p == 'type':
      if self.default is None:
        p = None
      else:
        p = type(self.default)  # cast value to type
    if p is not None:
      value = p(value)
    if self.enum and value not in self.enum:
      raise BadOptionValue(self, value)
    return value

  def FormatFlag(self):
    """Get the first flag for an self."""
    if self.flag is None:
      return ""
    if hasattr(self.flag, '__len__'):
      return SpecToFlag(self.flag[0])
    return SpecToFlag(self.flag)

  def FormatAllFlags(self):
    if self.flag is None:
      return ""
    if hasattr(self.flag, '__len__'):
      return ", ".join(map(SpecToFlag, self.flag))
    return SpecToFlag(self.flag)

# test get/set of unknown attrs raises AttributeError
# test del attr raises Exception
# test successful get/set of known attrs
# test construct from dict or from tuple of pairs

def SpecToFlag(flag):
  """Format an option's flag specification as a command-line switch.

  For example, a short flag like 'a' or 'a:' becomes '-a', while a long flag
  like 'aaa' or 'aaa=' becomes '--aaa'.

  """
  if flag[-1] in (':', '='):
    flag = flag[:-1]
  if len(flag) == 1:
    return '-' + flag
  return '--' + flag

class OptionError(Exception):
  """Indicates that an option was given a bad value."""

  def __init__(self, option = None, msg = None):
    if msg is None and isinstance(option, basestring):
      # allow message to be passed as first parameter
      msg = option
      option = None
    self.option = option
    self.msg = msg

  def __str__(self):
    if self.option is None:
      return self.msg or ""
    flag = self.option.FormatAllFlags()
    if flag:
      flag = " (%s)" % flag
    msg = self.msg
    if msg is None:
      msg = "option has bad value: %s" % self.option.value
    return "Option '%s'%s -- %s" % (self.option._name, flag, msg)

class BadOptionValue(OptionError):

  def __init__(self, option, value):
    super(BadOptionValue, self).__init__(option = option)
    self.supplied_value = value

  def __str__(self):
    flag = self.option.FormatAllFlags()
    if flag:
      flag = " (%s)" % flag
    return "Option '%s'%s had bad value: %s" % (self.option._name, flag,
        pformat(self.supplied_value))

class MissingOption(OptionError):

  def __init__(self, option):
    super(MissingOption, self).__init__(option = option)

  def __str__(self):
    flag = self.option.FormatAllFlags()
    if flag:
      flag = " (%s)" % flag
    return "Option '%s'%s is required" % (self.option._name, flag)

def IterateOptions(options):
  """Convert an option hierarchy to a flat list of options.

  :param options: Option hierarchy.
  :type options: Option or OptionGroup
  :rtype: list of Option

  """
  if isinstance(options, OptionGroup):
    values = list()
    for opt in options:
      values.extend(IterateOptions(opt))
    return values
  elif isinstance(options, Option):
    return [options]
  raise ValueError("Unknown option: %s" % options)

class OptData(object):
  def __init__(self, opt):
    self.option = opt
    short_flags = list()
    long_flags = list()
    flags = opt.flag
    if flags:
      if not hasattr(flags, '__len__'):
        flags = (flags,)
      # separate flags by type
      for flag in flags:
        if len(flag) == 1 or (len(flag) == 2 and flag[1] == ':'):
          short_flags.append(flag[:1])
        elif flag[-1] == '=':
          long_flags.append(flag[:-1])
        else:
          long_flags.append(flag)
    self.short_flags = short_flags
    self.long_flags = long_flags
  @property
  def doc(self):
    return self.option.doc
  @property
  def value(self):
    return self.option.value

def GetOptionData(options):
  data = list()
  for opt in IterateOptions(options):
    d = OptData()
    data.append(d)
  return data

def PrintUsage(options, stream=None, max_flag_length=25, width=None):
  """Print a help message for the given command line options.

  :param stream: File to which usage message is printed (default is sys.stderr).
  :param int max_flag_length: Maximum size of option flag string, after which a
     newline is printed before the option doc string.

  """
  if stream is None:
    stream = sys.stderr
  if width is None:
    try:
      # Try to read terminal width (ony for *nix systems)
      _,width = os.popen('stty size', 'r').read().split()
      width = int(width)
    except:
      width = 70

  # compare options by short flag, or first character of long flag
  def get(xs):
    return (xs and xs[0]) or ''
  def optcmp(x, y):
    x = get(x.short_flags) or get(x.long_flags)
    y = get(y.short_flags) or get(y.long_flags)
    result = cmp(x[:1].lower(), y[:1].lower())
    if result == 0:
      result = cmp(x, y)
    return result

  opt_data = map(OptData, IterateOptions(options))
  opt_data.sort(optcmp)
  for opt in opt_data:
    doc = ""
    if opt.doc is not None:
      doc += str(opt.doc)
    if opt.value is not None:
      doc += " (default: %s)" % pformat(opt.value)
    short_flags = ["-%s" % f for f in opt.short_flags]
    long_flags = ["--%s" % f for f in opt.long_flags]
    if short_flags:
      flag_str = ", ".join(short_flags + long_flags)
    else:
      flag_str = "    " + ", ".join(long_flags)
    templ = "%%-%ds" % max_flag_length
    print >>stream, templ % flag_str,
    if len(flag_str) <= max_flag_length:
      # Use initial indent to account for printed flags, but remove it afterward
      s = textwrap.fill(doc, width = width,
          subsequent_indent = ' ' * (max_flag_length + 3),
          initial_indent = ' ' * (max_flag_length + 3))
      s = s[max_flag_length + 3:]
      print >>stream, " ", s
    else:
      print >>stream, "\n%s" % textwrap.fill(doc, width = width,
          subsequent_indent = ' ' * (max_flag_length + 3),
          initial_indent = ' ' * (max_flag_length + 3))

def ParseCommandLine(options, argv = None):
  """Set the values of an option hierarchy from command-line arguments.

  :param options: Option hierarchy.
  :type options: Option or OptionGroup
  :param argv: Command-line arguments, or `sys.argv` if unset.
  :type argv: list of str
  :rtype: list of str
  :returns: Any arguments that were not parsed.

  """
  short_flags = ""
  long_flags = []
  opt_map = dict()
  has_arg_map = dict()  # indicates whether given option takes an argument
  for opt in IterateOptions(options):
    flags = opt.flag
    if flags is None: continue
    if not hasattr(flags, '__len__'):
      flags = (flags,)
    for flag in flags:
      has_arg = False
      if len(flag) == 1 or (len(flag) == 2 and flag[1] == ':'):
        short_flags += flag
        flag_str = '-' + flag[:1]
        if len(flag) > 1:
          has_arg = True
      else:
        long_flags += [flag]
        if flag[-1] == '=':
          flag_str = '--' + flag[:-1]
          has_arg = True
        else:
          flag_str = '--' + flag
      if flag_str in opt_map:
        raise Exception("The command-line flag '%s' " % flag_str +
            "is used by more than one option.")
      opt_map[flag_str] = opt
      has_arg_map[flag_str] = has_arg
  cli_opts, unparsed_args = GetOptions(short_flags, long_flags, argv=argv)
  for cli_opt, cli_arg in cli_opts:
    opt = opt_map[cli_opt]
    if has_arg_map[cli_opt]:
      arg = cli_arg
    else:
      arg = True
    opt.value = arg
  return unparsed_args

class UsageException(OptionError):
  """An exception indicating that a program was called inappropriately.

  For example, this could be thrown if a program was passed an invalid command
  line argument.

  """

def Usage(msg, exc = None):
  """Print the usage message for a program and exit.

  :param str msg: Usage message.
  :param exc: Exception information to include in printed message.

  """
  if exc and type(exc) == UsageException and exc.msg:
    print >>sys.stderr, exc.msg
  msg = "usage: %s %s" % (os.path.basename(sys.argv[0]), msg)
  sys.exit(msg)

def GetOptions(short_opts, long_opts = (), argv = None):
  """Parse command line arguments, raising a UsageException if an error is
  found.

  :param str short_opts: Set of single-character argument keys (see
     documentation for the :mod:`getopt` module).
  :param long_opts: Set of multi-character argument keys.
  :type long_opts: list of str
  :param argv: Command line arguments to parse. Defaults to :attr:`sys.argv`.
  :type argv: list of str
  :returns: Parsed options, and remaining (unparsed) arguments.
  :rtype: 2-tuple, where the first element is a list of pairs of str, and the
     second element is a list of str.

  """
  if argv == None:
    argv = sys.argv[1:]
  try:
    opts, args = getopt.getopt(argv, short_opts, long_opts)
  except getopt.GetoptError,e:
    raise UsageException(str(e))
  return opts, args
