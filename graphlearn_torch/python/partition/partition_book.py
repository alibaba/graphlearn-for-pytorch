import torch
from typing import List, Tuple
from .base import PartitionBook


class RangePartitionBook(PartitionBook):
  r"""A class for managing range-based partitions of consecutive IDs.
  Suitable when IDs within each partition are consecutive.
  Args:
      partition_ranges (List[Tuple[int, int]]): A list of tuples representing
          the start and end (exclusive) of each partition range.
      partition_idx (int): The index of the current partition.
  Example:
      >>> partition_ranges = [(0, 10), (10, 20), (20, 30)]
      >>> range_pb = RangePartitionBook(partition_ranges, partition_idx=1)
      >>> indices = torch.tensor([0, 5, 10, 15, 20, 25])
      >>> partition_ids = range_pb[indices]
      >>> print(partition_ids)
      tensor([0, 0, 1, 1, 2, 2])
  """

  def __init__(self, partition_ranges: List[Tuple[int, int]], partition_idx: int):
    if not all(r[0] < r[1] for r in partition_ranges):
      raise ValueError("All partition ranges must have start < end")
    if not all(r1[1] == r2[0] for r1, r2 in zip(partition_ranges[:-1], partition_ranges[1:])):
      raise ValueError("Partition ranges must be continuous")

    self.partition_bounds = torch.tensor(
        [end for _, end in partition_ranges], dtype=torch.long)
    self.partition_idx = partition_idx
    self._id2index = OffsetId2Index(partition_ranges[partition_idx][0])

  def __getitem__(self, indices: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(self.partition_bounds, indices, right=True)

  @property
  def device(self):
    return self.partition_bounds.device

  @property
  def id2index(self):
    return self._id2index

  def id_filter(self, node_pb: PartitionBook, partition_idx: int):
    start = self.partition_bounds[partition_idx-1] if partition_idx > 0 else 0
    end = self.partition_bounds[partition_idx]
    return torch.arange(start, end)


class OffsetId2Index:
  r"""
  Convert global IDs to local indices by subtracting a specified offset.
  """

  def __init__(self, offset: int):
    self.offset = offset

  def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
    local_indices = ids - self.offset
    return local_indices

  def to(self, device):
    # device is always same as the input ids
    return self


class GLTPartitionBook(PartitionBook, torch.Tensor):
  r""" A partition book of graph nodes or edges.
  """

  def __getitem__(self, indices) -> torch.Tensor:
    return torch.Tensor.__getitem__(self, indices)
