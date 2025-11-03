from typing import Optional

from datasets import Dataset as HFDataset  # type: ignore

from .hotpotqa import load_hotpotqa


def get_dataset(
    name: str,
    setting: str,
    split: str,
    source: str = "auto",
    limit: Optional[int] = None,
) -> HFDataset:
    """Dispatch to dataset-specific loaders and return a unified HF Dataset."""
    name_norm = name.lower()
    if name_norm == "hotpotqa":
        return load_hotpotqa(setting=setting, split=split, source=source, limit=limit)
    raise ValueError(f"Unsupported dataset: {name}")


__all__ = ["get_dataset", "load_hotpotqa"]


