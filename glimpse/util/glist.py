"""Miscellaneous functions that do not belong in one of the other modules."""

# Copyright (c) 2011-2013 Mick Thomure
# All rights reserved.
#
# Please see the file LICENSE.txt in this distribution for usage terms.

import itertools

def TakePairs(lst):
  """Convert a list of values into a list of 2-tuples.

  Example usage::

     >>> TakePairs([1, 2, 3, 4])
     [(1, 2), (3, 4)]

  """
  return [ (lst[i],lst[i+1]) for i in range(0, len(lst), 2) ]

def TakeTriples(lst):
  """Convert a list of values into a list of 3-tuples.

  Example usage::

     >>> TakeTriples([1, 2, 3, 4, 5])
     [(1, 2, 3), (4, 5)]

  """
  return [ (lst[i],lst[i+1],lst[i+2]) for i in range(0, len(lst), 3) ]

def GroupIterator(elements, group_size):
  """Create an iterator that returns sub-groups of an underlying iterator.

  For example, using a group size of three with an input of the first seven
  natural numbers will result in the elements (0, 1, 2), (3, 4, 5), (6,). Note
  that tail elements are not ignored.

  :param elements: An iterable sequence of input values.
  :param int group_size: Number of elements in each returned sub-group.
  :returns: Iterator over sub-groups.

  Example usage::

     >>> GroupIterator([1, 2, 3, 4, 5, 6, 7, 8], 3)
     [(1, 2, 3), (4, 5, 6), (7, 8)]

  """
  element_iter = iter(elements)
  while True:
    batch = tuple(itertools.islice(element_iter, group_size))
    if len(batch) == 0:
      raise StopIteration
    yield batch

def UngroupIterator(groups):
  """Create an iterator that returns each element from each group, one element
  at a time.

  This is the inverse of :func:`GroupIterator`.

  :param groups: A list of iterators, where each iterator may have a different
     lengths.
  :returns: A single iterator that returns all the elements from the first input
     iterator, then all the elements of the second iterator, and so on.

  Example usage::

     >>> UngroupIterator([(1, 2, 3), (4,), (5, 6, 7, 8)])
     [1, 2, 3, 4, 5, 6, 7, 8]

  """
  return itertools.chain(*groups)

def UngroupLists(groups):
  """Concatenate several sequences to form a single list.

  :rtype: list

  .. seealso::
     :func:`UngroupIterator`.

  """
  return list(itertools.chain(*groups))

def SplitList(data, sizes = []):
  """Break a list into unequal-sized sublists.

  :param list data: Input data.
  :param sizes: Size of each chunk. If sum of sizes is less than entire size of
     input array, the remaining elements are returned as an extra sublist in the
     result.
  :type sizes: list of int
  :returns: Sublists of requested size.
  :rtype: list of list

  .. seealso::
     :func:`UngroupIterator`

  """
  assert(all([ s >= 0 for s in sizes ]))
  if len(sizes) == 0:
    return data
  if sum(sizes) < len(data):
    sizes = list(sizes)
    sizes.append(len(data) - sum(sizes))
  out = list()
  last = 0
  for s in sizes:
    out.append(data[last : last+s])
    last += s
  return out
