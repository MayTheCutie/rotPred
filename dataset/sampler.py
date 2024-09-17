from typing import Iterator, List, Optional, Union
from operator import itemgetter
from torch.utils.data import Dataset, Sampler, DistributedSampler
import torch
import math

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

# from https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py#L499
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class DynamicRangeSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        try:
            self.dataset.update_filtered_indices()
        except AttributeError: # Subset
            self.dataset.dataset.update_filtered_indices()

    def __iter__(self):
        try:
            return iter(self.dataset.filtered_indices)
        except AttributeError: # Subset
            return iter(self.dataset.dataset.filtered_indices)

    def __len__(self):
        try:
            return len(self.dataset.filtered_indices)
        except AttributeError:
            return len(self.dataset.dataset.filtered_indices)
        
class DynamicRangeDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.dataset = dataset
        self.epoch = 0

    def __iter__(self):
        # Ensure filtered_indices are up to date
        if hasattr(self.dataset, 'update_filtered_indices'):
            self.dataset.update_filtered_indices()
        elif hasattr(self.dataset.dataset, 'update_filtered_indices'):
            self.dataset.dataset.update_filtered_indices()

        # Get the current filtered indices
        if hasattr(self.dataset, 'filtered_indices'):
            filtered_indices = self.dataset.filtered_indices
        elif hasattr(self.dataset.dataset, 'filtered_indices'):
            filtered_indices = self.dataset.dataset.filtered_indices
        else:
            filtered_indices = list(range(len(self.dataset)))

        # Shuffle if needed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(filtered_indices), generator=g).tolist()
        else:
            indices = list(range(len(filtered_indices)))

        # Ensure the number of samples is divisible by num_replicas
        self.num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # Add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # Map indices back to the filtered indices
        return iter(filtered_indices[i] for i in indices)

    def __len__(self):
        if hasattr(self.dataset, 'filtered_indices'):
            return len(self.dataset.filtered_indices)
        elif hasattr(self.dataset.dataset, 'filtered_indices'):
            return len(self.dataset.dataset.filtered_indices)
        else:
            return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch
        print("Sampler: epoch was set to ", epoch)
        if hasattr(self.dataset, 'expand_label_range'):
            self.dataset.expand_label_range()
        elif hasattr(self.dataset.dataset, 'expand_label_range'):
            self.dataset.dataset.expand_label_range()