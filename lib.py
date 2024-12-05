from collections import defaultdict, deque, Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import functools
from functools import cache
import heapq
import itertools
from itertools import permutations
import re
import numpy as np
from numpy.polynomial import Polynomial
from typing import Any

EN_TO_NUM = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
}

# Directions for grid traversal
N = np.array([-1, 0])
E = np.array([0, 1])
S = np.array([1, 0])
W = np.array([0, -1])
ORTHOGONAL_DIRS = [N, E, S, W]
DIAGONAL_DIRS = [N + E, N + W, S + E, S + W]
ADJ_DIRS = ORTHOGONAL_DIRS + DIAGONAL_DIRS
UP = N
DOWN = S
LEFT = W
RIGHT = E


def valid_index(lst: Sequence[Any], index: int) -> bool:
  if isinstance(index, int):
    if index < 0 or index >= len(lst):
      return False
    return lst[index]
  if isinstance(index, Iterable):
    for i in index:
      if i < 0 or i >= len(lst):
        return False
      lst = lst[i]
    return True
  raise ValueError('index must be of type int or Iterable[int].')


def get(lst: Sequence[Any], index: int, default: Any) -> Any:
  if isinstance(index, int):
    if index < 0 or index >= len(lst):
      return default
    return lst[index]
  if isinstance(index, Iterable):
    for i in index:
      if i < 0 or i >= len(lst):
        return default
      lst = lst[i]
    return lst
  raise ValueError('index must be of type int or Iterable[int].')


def groups_of(num: int, iterable: Iterable[Any]) -> list[list[Any]]:
  """Splits iterable into groups of num."""
  groups = []
  group = []
  for x in iterable:
    group.append(x)
    if len(group) >= num:
      groups.append(group)
      group = []
  if group:
    print(f'Warning: Total number of elements not divisible by {num}!')
    groups.append(group)
  return groups


def int_round(x: float) -> int:
  rounded = np.round(x)
  if isinstance(x, Iterable):
    return rounded.astype(int)
  return int(rounded)


# Deprecated: bad version of collections.Counter, created before I knew it existed
def multiset(iterable: Iterable[Any], keys: Iterable[Any] | None = None) -> dict[Any, int]:
  counts = defaultdict(lambda: 0)
  for x in iterable:
    counts[x] += 1
  if keys is None:
    return counts
  return {k: counts.get(k, 0) for k in keys}


def num_to_base(num: int, base: int, fixed_len: int | None = None) -> str:
  radices = []
  while num > 0:
    radices.append(num % base)
    num //= base
  if fixed_len is None:
    return list(reversed(radices))
  if len(radices) > fixed_len:
    print(f'{num} in base {base} is longer than given fixed length {fixed_len}.')
  prefix = [0 for _ in range(fixed_len - len(radices))]
  return prefix + radices

def powerset(iterable: Iterable[Any]) -> Iterable[tuple[Any]]:
  'powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)'
  s = list(iterable)
  return itertools.chain.from_iterable(
      itertools.combinations(s, r) for r in range(len(s) + 1)
  )


def process_raw(raw: str, num_to_print: int = 10) -> list[str] | list[list[str]]:
  """Processes raw AoC data."""
  if '\n\n' in raw:
    print('Splitting into blocks...')
    blocks = raw.split('\n\n')
    print(f'{len(blocks)} blocks found.')
    liness = [block.splitlines() for block in blocks]
    if not liness:
      return liness

    uniform = True
    first_len = len(liness[0])
    for lines in liness:
      if len(lines) != first_len:
        uniform = False
    if uniform:
      print(f'Each block has {first_len} lines.')
      return liness

    if len(liness) < num_to_print:
      print('The blocks have the following number of lines:')
    else:
      print(
          f'The first {num_to_print} blocks have the following number of lines:'
      )
    for i, lines in enumerate(liness):
      print(f'  {len(lines)}')
      if i >= num_to_print:
        break
    return liness

  print('Splitting into lines...')
  lines = raw.splitlines()
  print(f'{len(lines)} lines found.')
  uniform = True
  first_len = len(lines[0])
  for line in lines:
    if len(lines) != first_len:
      uniform = False
      break
  if uniform:
    print(f'{first_len} characters in each line.')
    return lines

  max_len = 0
  min_len = float('inf')
  for line in lines:
    max_len = max(len(line), max_len)
    min_len = min(len(line), min_len)
  print(f'{min_len} characters in the shortest line.')
  print(f'{max_len} characters in the longest line.')
  return lines


def range_intersection(ranges1: list[Sequence[int, int]], ranges2: list[Sequence[int, int]]) -> list[Sequence[int, int]]:
  i1 = 0
  i2 = 0
  result = []
  ranges1 = list(sorted(ranges1, key=lambda r: r[0]))
  ranges2 = list(sorted(ranges2, key=lambda r: r[0]))
  while i1 < len(ranges1) and i2 < len(ranges2):
    r1 = ranges1[i1]
    r2 = ranges2[i2]
    result.append((max(r1[0], r2[0]), min(r1[1], r2[1])))
    if r1[1] < r2[1]:
      i1 += 1
    else:
      i2 += 1
  return range_merge(result)


def range_merge(ranges: Iterable[Sequence[int, int]]) -> list[Sequence[int, int]]:
  result = []
  for r in sorted(ranges, key=lambda r: r[0]):
    if r[0] >= r[1]:
      continue
    if result and result[-1][1] >= r[0]:
      result[-1] = result[-1][0], max(r[1], result[-1][1])
      continue
    result.append(r)
  return result


def range_union(ranges1: list[Sequence[int, int]], ranges2: list[Sequence[int, int]]) -> list[Sequence[int, int]]:
  return range_merge(ranges1 + ranges2)
