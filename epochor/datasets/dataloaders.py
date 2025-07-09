import time
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from typing import Iterator, Optional, Any, Dict

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


class StaticSyntheticDataset(TorchDataset):
    def __init__(self, data_dict):
        # Infer length and ensure all fields match
        self.length = len(next(iter(data_dict.values())))
        for k, v in data_dict.items():
            if len(v) != self.length:
                raise ValueError(f"Length mismatch in key '{k}'")

        # Store data as-is, but convert numeric fields to tensors
        self.data = {
            k: torch.as_tensor(v) if self._is_tensor_like(v[0]) else v
            for k, v in data_dict.items()
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    @staticmethod
    def _is_tensor_like(x):
        return isinstance(x, (int, float, bool)) or torch.is_tensor(x)


class SyntheticIterableDataset(IterableDataset):
    def __init__(self, data_dict):
        self.data = data_dict
        self.length = len(next(iter(data_dict.values())))

    def __iter__(self):
        for i in range(self.length):
            yield {k: self.data[k][i] for k in self.data}

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
                data =  bench.prepare_data(seed=seed)

                dataset = StaticSyntheticDataset(
                    StaticDictDataset(data), 
                    batch_size=16)
                
                # SyntheticTimeSeriesDataset yields full batches, so batch_size=None for the DataLoader.
                return DataLoader(dataset, batch_size=None, num_workers=0)
            case _:
                raise NotImplementedError(f"Dataset {dataset_id} not implemented.")
