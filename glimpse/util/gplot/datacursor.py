# From: http://stackoverflow.com/questions/4652439/is-there-a-matplotlib-equivalent-of-matlabs-datacursormode/4674445#4674445

from matplotlib import cbook
import numpy as np

def _update_dict(old, new):
  """Update the elements of a (possibly nested) dictionary."""
  if new is None:
    return
  for k,v in new.items():
    if hasattr(v, 'items') and k in old:
      _update_dict(old[k], v)
    else:
      old[k] = v

class DataCursor(object):
    """A simple data cursor widget that displays the x,y location of a
    matplotlib artist when it is selected."""
    def __init__(self, artists, tolerance=5, offsets=(-20, 20),
                 template='x: %0.2f\ny: %0.2f', display_all=False,
                 annotate_opts = None):
        """Create the data cursor and connect it to the relevant figure.
        "artists" is the matplotlib artist or sequence of artists that will be
            selected.
        "tolerance" is the radius (in points) that the mouse click must be
            within to select the artist.
        "offsets" is a tuple of (x,y) offsets in points from the selected
            point to the displayed annotation box
        "template" is the format string to be used. Note: For compatibility
            with older versions of python, this uses the old-style (%)
            formatting specification.
        "display_all" controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless.
        "annotate_opts" contains keyword options that are passed to the
            annotate() function for each axis.
        """
        if isinstance(template, basestring):
          template_ = lambda x, y, i: template % (x, y)
        else:
          template_ = template
        self.template = template_
        self.offsets = offsets
        self.display_all = display_all
        if not cbook.iterable(artists):
            artists = [artists]
        self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))
        self.annotations = {}
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax, annotate_opts)
        for artist in self.artists:
            artist.set_picker(tolerance)
        for fig in self.figures:
            fig.canvas.mpl_connect('pick_event', self)

    def annotate(self, ax, opts):
        """Draws and hides the annotation box for the given axis "ax"."""
        opts_ = dict(ha='right',
                xytext=self.offsets, textcoords='offset points', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        _update_dict(opts_, opts)
        annotation = ax.annotate("", xy=(0, 0), **opts_)
        annotation.set_visible(False)
        return annotation

    def __call__(self, event):
        """Intended to be called through "mpl_connect"."""
        # Rather than trying to interpolate, just display the clicked coords
        # This will only be called if it's within "tolerance", anyway.
        if event.mouseevent.xdata is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in self.annotations.values():
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            self.update_annotation(event)
        event.canvas.draw()

    def get_point(self, event):
        import matplotlib.patches
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        artist = event.artist
        if not hasattr(artist, 'properties'):
            return x, y, None
        props = artist.properties()
        if isinstance(artist, matplotlib.patches.Rectangle):
            # bar plot
            try:
              idx = self.artists.index(artist)
            except ValueError:
              return x, y, None
            x = props['x'] + props['width'] / 2.
            y = props['y'] + props['height']
        else:
            if 'xydata' in props:
                # line plot
                data = props['xydata']
            elif 'offsets' in props:
                # scatter plot
                data = props['offsets']
            else:
                return x, y, None
            # find the closest datapoint
            idx = np.argmin(np.sqrt((data[:,0] - x)**2 + (data[:,1] - y)**2))
            x, y = data[idx]
        return x, y, idx

    def update_annotation(self, event):
        x, y, i = self.get_point(event)
        text = self.template(x, y, i)
        if not text:
          return
        annotation = self.annotations[event.artist.axes]
        annotation.xy = x, y
        annotation.set_text(text)
        annotation.set_visible(True)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2,1,1)
    line1, = plt.plot(range(10), 'ro-')
    plt.subplot(2,1,2)
    line2, = plt.plot(range(10), 'bo-')
    DataCursor([line1, line2])
    plt.show()
