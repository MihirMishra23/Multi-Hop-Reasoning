from typing import Optional

from datasets import Dataset as HFDataset  # type: ignore

from .hotpotqa import load_hotpotqa
from .musique import load_musique
from .mquake import load_mquake


def get_dataset(
    name: str,
    split: str,
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    setting: str | None = None,
) -> HFDataset:
    """Dispatch to dataset-specific loaders and return a unified HF Dataset."""
    name_norm = name.lower()
    if name_norm == "hotpotqa":
        return load_hotpotqa(setting=setting, split=split, source=source, limit=limit, seed=seed)
    if name_norm == "musique":
        # MuSiQue has no 'setting'; ignore the argument
        return load_musique(split=split, source=source, limit=limit, seed=seed)
    if name_norm == "mquake-remastered":
        return load_mquake(split = split, limit = limit, seed = seed)
    raise ValueError(f"Unsupported dataset: {name}")


__all__ = ["get_dataset", "load_hotpotqa", "load_musique"]
