
class IExecutor:
  """An executor is an interface for interacting with a set of task processors.
  This interface includes a request queue (of which only the Put() method is
  exposed), and a result queue (of which only the Get() method is exposed)."""

  def Put(self, request):
    """Submit a task to the executor, whose result can be obtained by calling
    Get(). Each reques submitted via Put() is guaranteed to produce exactly one
    result via Get()."""

  def PutMany(self, requests):
    """Submit a batch of multiple requests. Some executors may have better
    performance in this case, as compared to submitting multiple individual
    requests.
    RETURN (int) the number of submitted requests"""

  def Get(self):
    """Retrieve the result for a request submitted via Put()."""

  def GetMany(self, num_results):
    """Retrieve a batch of results. As with PutMany(), some executors may have
    better performance using this method as compared to repeatedly calling
    Get()."""

  def IsEmpty(self):
    """Determine if any results are available via Get()."""

def ExecutorMap(executor, requests):
  """Apply the operation implemented by an executor across an iterable of
  requests. Note that the order of return elements may not match that of the
  input arguments (depending on implementation), and that an empty queue is
  assumed."""
  assert executor.IsEmpty(), "Map behavior is undefined unless executor " \
      "queue is empty."
  num_requests = executor.PutMany(requests)
  return executor.GetMany(num_requests)

class BasicExecutor:
  """A simple executor that applies the callback to each request in the same
  thread."""

  def __init__(self, callback):
    self.callback = callback
    self.queue = []

  def Put(self, request):
    self.queue.append(self.callback(request))

  def PutMany(self, requests):
    return len(map(self.Put, requests))

  def Get(self):
    """Retrieve the result of running a task submitted via Put()."""
    return self.queue.pop()

  def GetMany(self, num_results):
    """Retrieve a batch of multiple results. As with PutMany(), some executors
    may have better performance using this methdo as compared to repeatedly
    calling Get()."""
    return ( self.Get() for _ in range(num_results) )

  def IsEmpty(self):
    """Determine if any elements are available via Get()."""
    return len(self.queue) == 0
