from datasets import load_dataset, concatenate_datasets, Dataset
import pyarrow.parquet as pq
import random

_MQUAKE_PARQUET = "/share/j_sun/lmlm_multihop/mquake/CF6334-00000-of-00001.parquet"
_SPLIT_KEY = "6334"  # key inside the 'split' struct for the no-conflict 6334 subset


def _get_labels(example):
    split_col = example.get("split") or {}
    return split_col.get(_SPLIT_KEY) or []


def _rename_question(example):
    example["question"] = random.choice(example["questions"])
    # Original answers (for evaluating with original database)
    example["answers"] = [example["answer"]] + example["answer_alias"]
    # New answers (for evaluating with edited database)
    example["new_answers"] = [example["new_answer"]] + example["new_answer_alias"]
    # Tag each example with its MQuAKE split type for per-type metric reporting
    labels = _get_labels(example)
    if "test_edited" in labels:
        example["mquake_split_type"] = "test_edited"
    elif "train_edited" in labels:
        example["mquake_split_type"] = "train_edited"
    else:
        example["mquake_split_type"] = "test_unedited"
    return example


def load_mquake(split: str, limit: int, seed: int = 42):
    # Load via pyarrow; strip embedded HF metadata which has a broken schema definition
    pa_table = pq.read_table(_MQUAKE_PARQUET)
    pa_table = pa_table.replace_schema_metadata(None)
    raw_dataset = Dataset(pa_table)

    raw_dataset = raw_dataset.shuffle(seed=seed)

    # Use label-based selection so test_edited / train_edited / test_unedited are correctly separated
    if split == "train":
        subset = raw_dataset.filter(lambda ex: "train_edited" in _get_labels(ex))
    elif split == "test" or split.startswith("eval-"):
        # Include both edited and unedited test examples so per-type metrics can be computed
        subset = raw_dataset.filter(
            lambda ex: bool({"test_edited", "test_unedited"} & set(_get_labels(ex)))
        )
    else:
        subset = raw_dataset

    subset = subset.map(_rename_question, load_from_cache_file=False)

    if limit is None:
        limit = len(subset)

    return subset.select(range(min(limit, len(subset))))
