import time
import torch
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from typing import Iterator, Tuple, Optional, Any, Dict

from epochor.generators import CombinedGenerator
from epochor.datasets.ids import DatasetId
from epochor.generators.synthetic_v1 import SyntheticBenchmarkerV1

# Attempt to import HuggingFace streaming IterableDataset
try:
    from datasets import IterableDataset as HfIterableDataset
except ImportError:
    HfIterableDataset = None


class SyntheticTimeSeriesDataset(TorchIterableDataset):
    """
    Wraps SyntheticBenchmarkerV1 so that each iteration yields one batch of synthetic data.
    Each call to `bench.prepare_data(seed)` returns a dict containing:
      - "inputs_padded": Tensor (n_series, max_input_len)
      - "attention_mask": Tensor (n_series, max_input_len)
      - "targets_padded": Tensor (n_series, max_target_len)
      - "actual_target_lengths": List[int] of length n_series
    """

    def __init__(
        self,
        bench,
        *,
        start_seed: int = 0,
        num_batches: int = None
    ):
        """
        Args:
            bench:        Instance of SyntheticBenchmarkerV1.
            start_seed:   Integer seed for the first batch; subsequent batches use start_seed + batch_index.
            num_batches:  If provided, stops after yielding `num_batches` batches. If None, yields indefinitely.
        """
        super().__init__()
        self.bench = bench
        self.start_seed = start_seed
        self.num_batches = num_batches

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        batch_idx = 0
        while True:
            if self.num_batches is not None and batch_idx >= self.num_batches:
                break

            seed = self.start_seed + batch_idx
            batch_dict = self.bench.prepare_data(seed=seed)
            yield batch_dict

            batch_idx += 1


class TimeSeriesIterableDataset(TorchIterableDataset):
    """
    Wraps a CombinedGenerator so that each iteration yields (series, tag).
    `seed_fn` produces the seed (e.g., current_block) for each call.
    """

    def __init__(
        self,
        generator: CombinedGenerator,
        seed_fn: Optional[callable] = None,
    ):
        """
        Args:
            generator: CombinedGenerator instance
            seed_fn:   Function returning an int seed (e.g., `lambda: int(time.time())`).
                       If None, defaults to current time.
        """
        super().__init__()
        self.generator = generator
        self.seed_fn = (lambda: int(time.time())) if seed_fn is None else seed_fn

    def __iter__(self) -> Iterator[Tuple]:
        while True:
            seed = self.seed_fn()
            series, tag = self.generator.generate(seed=seed)
            yield series, tag


class DataLoaderFactory:
    """
    Produces PyTorch DataLoaders for a variety of source types:
      • CombinedGenerator → wraps in TimeSeriesIterableDataset
      • SyntheticTimeSeriesDataset → directly used
      • HuggingFace streaming IterableDataset → wrapped in DataLoader
      • torch.utils.data.IterableDataset (custom) → wrapped in DataLoader
      • torch.utils.data.Dataset → standard DataLoader
      • Existing DataLoader → returned as-is
    """

    @staticmethod
    def get_loader(
        source: Any,
        *,
        n_series,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        shuffle: bool = False,
        seed_fn: Optional[callable] = None,
    ) -> DataLoader:
        """
        Builds a DataLoader for the given `source`.

        Args:
            source:      One of:
                           - CombinedGenerator instance
                           - SyntheticTimeSeriesDataset instance
                           - HuggingFace streaming IterableDataset
                           - torch.utils.data.IterableDataset
                           - torch.utils.data.Dataset
                           - torch.utils.data.DataLoader (returned as-is)
            batch_size:  Batch size for DataLoader. Required if `source` is
                         SyntheticTimeSeriesDataset, HF streaming, or torch Dataset.
                         If `source` is CombinedGenerator, yields one (series, tag) at a time.
            num_workers: Number of worker processes (for Dataset or streaming).
            shuffle:     Whether to shuffle (only for regular Dataset).
            seed_fn:     If `source` is CombinedGenerator, function returning int seed.

        Returns:
            A torch.utils.data.DataLoader instance that yields:
              - (series, tag) if `source` is CombinedGenerator
              - dict of tensors if `source` is SyntheticTimeSeriesDataset
              - items from HF streaming IterableDataset if source is streaming
              - items from custom IterableDataset if source is IterableDataset
              - batched items from Dataset if source is Dataset
              - items from existing DataLoader if source is already one
        """
        # 1) If source is already a DataLoader, return it unchanged
        if isinstance(source, DataLoader):
            return source


        # 3) If source is our SyntheticTimeSeriesDataset
        if isinstance(source, SyntheticTimeSeriesDataset):
            if batch_size is not None:
                raise ValueError(
                    "SyntheticTimeSeriesDataset yields full batches internally; set batch_size=None."
                )
            return DataLoader(
                source,
                batch_size=None,
                num_workers=num_workers,
                shuffle=False,
            )

        # 4) If source is a HuggingFace streaming IterableDataset
        if HfIterableDataset is not None and isinstance(source, HfIterableDataset):
            if batch_size is None:
                raise ValueError(
                    "batch_size must be provided when `source` is a HuggingFace streaming IterableDataset."
                )
            return DataLoader(
                source,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )

        # 5) If source is any other Torch IterableDataset
        if isinstance(source, TorchIterableDataset):
            if batch_size is None:
                raise ValueError(
                    "batch_size must be provided when `source` is a torch IterableDataset."
                )
            return DataLoader(
                source,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )

        # 6) If source is a Torch Dataset (map-style), wrap in DataLoader
        if isinstance(source, TorchDataset):
            if batch_size is None:
                raise ValueError(
                    "batch_size must be provided when `source` is a torch Dataset."
                )
            return DataLoader(
                source,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )

        raise TypeError(
            f"Unsupported source type for DataLoaderFactory: {type(source)}"
        )

class DatasetLoaderFactory:
    @staticmethod
    def get_loader(
        dataset_id: DatasetId,
        dataset_kwargs: Dict[str, Any],
        seed: int,
    ) -> DataLoader:
        """Loads data samples from the appropriate dataset."""

        match dataset_id:
            case DatasetId.UNIVARIATE_SYNTHETIC:
                length = dataset_kwargs.get("sequence_length")
                n_series = dataset_kwargs.get("n_series")
                if length is None or n_series is None:
                    raise ValueError("dataset_kwargs must contain 'length' and 'n_series' for UNIVARIATE_SYNTHETIC")
                
                bench = SyntheticBenchmarkerV1(length=length, n_series=n_series)
                
                num_batches = dataset_kwargs.get("num_batches") # Optional

                dataset = SyntheticTimeSeriesDataset(
                    bench=bench,
                    start_seed=seed,
                    num_batches=num_batches
                )
                
                # SyntheticTimeSeriesDataset yields full batches, so batch_size=None for the DataLoader.
                return DataLoader(dataset, batch_size=None, num_workers=0)
            case _:
                raise NotImplementedError(f"Dataset {dataset_id} not implemented.")
