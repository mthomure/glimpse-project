
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import Tkinter
from Tkinter import N, S, E, W, NW, HORIZONTAL, LEFT, SOLID

# Copied from http://www.voidspace.org.uk/python/weblog/arch_d7_2006_07_01.shtml
class ToolTip(object):

  def __init__(self, widget):
    self.widget = widget
    self.tipwindow = None
    self.id = None
    self.x = self.y = 0

  def showtip(self, text):
    "Display text in tooltip window"
    self.text = text
    if self.tipwindow or not self.text:
      return
    x, y, cx, cy = self.widget.bbox("insert")
    x = x + self.widget.winfo_rootx() + 27
    y = y + cy + self.widget.winfo_rooty() +27
    self.tipwindow = tw = Tkinter.Toplevel(self.widget)
    tw.wm_overrideredirect(1)
    tw.wm_geometry("+%d+%d" % (x, y))
    try:
      # For Mac OS
      tw.tk.call("::tk::unsupported::MacWindowStyle",
                 "style", tw._w,
                 "help", "noActivates")
    except Tkinter.TclError:
      pass
    label = Tkinter.Label(tw, text = self.text, justify = LEFT,
                  background = "#ffffe0", relief = SOLID, borderwidth = 1,
                  font=("tahoma", "8", "normal"))
    label.pack(ipadx=1)

  def hidetip(self):
    tw = self.tipwindow
    self.tipwindow = None
    if tw:
      tw.destroy()

def createToolTip(widget, text):
  toolTip = ToolTip(widget)
  def enter(event):
    toolTip.showtip(text)
  def leave(event):
    toolTip.hidetip()
  widget.bind('<Enter>', enter)
  widget.bind('<Leave>', leave)

# Copied from: http://effbot.org/zone/tkinter-autoscrollbar.htm
class AutoScrollbar(Tkinter.Scrollbar):

  # a scrollbar that hides itself if it's not needed.  only
  # works if you use the grid geometry manager.
  def set(self, lo, hi):
    if float(lo) <= 0.0 and float(hi) >= 1.0:
      # grid_remove is currently missing from Tkinter!
      self.tk.call("grid", "remove", self)
    else:
      self.grid()
    Tkinter.Scrollbar.set(self, lo, hi)

  def pack(self, **kw):
    raise Tkinter.TclError, "cannot use pack with this widget"

  def place(self, **kw):
    raise Tkinter.TclError, "cannot use place with this widget"

# Copied from: http://effbot.org/zone/tkinter-autoscrollbar.htm
def MakeScrollFrame(root, make_contents_callback, **args):
  vscroll = AutoScrollbar(root)
  vscroll.grid(row = 0, column = 1, sticky = N+S)
  hscroll = AutoScrollbar(root, orient = HORIZONTAL)
  hscroll.grid(row = 1, column = 0, sticky = E+W)
  canvas = Tkinter.Canvas(root, yscrollcommand = vscroll.set,
      xscrollcommand = hscroll.set, **args)
  canvas.grid(row = 0, column = 0, sticky = N+S+E+W)
  vscroll.config(command = canvas.yview)
  hscroll.config(command = canvas.xview)
  # make the canvas exapandable
  root.grid_rowconfigure(0, weight = 1)
  root.grid_columnconfigure(0, weight = 1)
  frame = make_contents_callback(canvas)
  frame.rowconfigure(1, weight = 1)
  frame.columnconfigure(1, weight = 1)
  canvas.create_window(0, 0, anchor = NW, window = frame)
  frame.update_idletasks()
  canvas.config(scrollregion = canvas.bbox("all"))
  return canvas, frame

class Converter(object):

  def __init__(self, convert_func):
    self.convert_func = convert_func

  def Check(self, x):
    try:
      x = self.convert_func(x)
    except:
      return False
    return True

  def Convert(self, x):
    return self.convert_func(x)

class MyEntry(Tkinter.Entry):

  def __init__(self, master, value = None, checker = None, setter = None,
      **args):
    self.last_value = value
    self.checker = checker
    self.setter = setter
    self.var = Tkinter.StringVar()
    self.var.set(value)
    args['textvariable'] = self.var
    args['validatecommand'] = (
        master.register(self._validate),  # make new Tcl command
        '%P',  # pass new entry value
        )
    args['validate'] = 'focusout'
    Tkinter.Entry.__init__(self, master, **args)

  def _validate(self, new):
    """Intended to be used as Tk callback only."""
    if not self.checker or self.checker(new):
      self.last_value = new
      if self.setter != None:
        self.setter(new)
    else:
      self.var.set(self.last_value)
      self.configure(validate = 'focusout')  # have to reset this for some
                                             # reason, or else validation only
                                             # works once
      return False
    return True

  def get(self):
    return self.var.get()

  def set(self, x):
    self.var.set(x)
    self.configure(validate = 'focusout')  # have to reset this for some
                                           # reason, or else validation only
                                           # works once

class MyCheckbutton(Tkinter.Checkbutton):

  def __init__(self, master = None, value = None, setter = None, **args):
    self.var = Tkinter.IntVar(master)
    self.var.set(value)
    self.setter = setter
    self.checker = Converter(bool).Check
    args['variable'] = self.var
    args['command'] = master.register(self._setter)
    Tkinter.Checkbutton.__init__(self, master, **args)

  def _setter(self):
    if self.setter != None:
      self.setter(self.var.get())

  def get(self):
    return self.var.get()

  def set(self, x):
    if self.checker != None and not self.checker(x):
      raise ValueError("Value did not pass validation")
    self.last_value = x
    self.var.set(x)
