from torch.utils.data import DataLoader
from torch.utils.data import Dataset as Dataset
from typing import Any, Dict
import torch
from epochor.datasets.ids import DatasetId
from epochor.generators.synthetic_v1 import SyntheticBenchmarkerV1


class StaticSyntheticDataset(Dataset):
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
                length = dataset_kwargs.get("length") # Changed from "sequence_length" to "length"
                n_series = dataset_kwargs.get("n_series")
                if length is None or n_series is None:
                    raise ValueError("dataset_kwargs must contain 'length' and 'n_series' for UNIVARIATE_SYNTHETIC")
                
                bench = SyntheticBenchmarkerV1(length=length, n_series=n_series)
                
                num_batches = dataset_kwargs.get("num_batches") # Optional
                data = bench.prepare_data(seed=seed)
                
                # Corrected: Directly pass the data dictionary to StaticSyntheticDataset
                # and extract batch_size from dataset_kwargs
                batch_size = dataset_kwargs.get("batch_size", 16) # Default to 16 if not provided
                dataset = StaticSyntheticDataset(data)
                
                # Use a regular DataLoader with the specified batch_size
                return DataLoader(dataset, batch_size=batch_size, num_workers=0)
            case _:
                raise NotImplementedError(f"Dataset {dataset_id} not implemented.")
