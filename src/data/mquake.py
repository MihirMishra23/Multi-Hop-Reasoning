from datasets import load_dataset, concatenate_datasets, Dataset
import pyarrow.parquet as pq
import random

def _rename_question(example):
    example["question"] = random.choice(example["questions"])
    # Original answers (for evaluating with original database)
    example["answers"] = [example["answer"]] + example["answer_alias"]
    # New answers (for evaluating with edited database)
    example["new_answers"] = [example["new_answer"]] + example["new_answer_alias"]
    # BUG: to solve the database contamination issue, we need to set the new_answers according to the edit tag
    # but we've already done _filter_example
    return example


def _filter_example(example):
    labels = example['split']['ALL']
    return 'train_edited' in labels or 'test_edited' in labels 


def load_mquake(split : str, limit : int, seed : int = 42):
    # Load via pyarrow to bypass broken HF metadata embedded in the parquet file
    pa_table = pq.read_table("/share/j_sun/lmlm_multihop/mquake/CF6334-00000-of-00001.parquet")
    raw_dataset = Dataset(pa_table)

    no_conflict_6334_subset  = raw_dataset.filter(_filter_example)
    no_conflict_6334_subset = no_conflict_6334_subset.shuffle(seed = seed)
    # See code examples here for selecting the dataset: https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered
    # This selects the 6334 subset which we can freely edit without contamination.

    if limit is None:
        limit = len(no_conflict_6334_subset)
        
    if split == 'train':
        no_conflict_6334_subset = no_conflict_6334_subset.select(range(5334))

    if split == 'test' or split.startswith('eval-'):
        no_conflict_6334_subset = no_conflict_6334_subset.select(range(5334, 6334))
    no_conflict_6334_subset = no_conflict_6334_subset.map(_rename_question, load_from_cache_file=False)

    return no_conflict_6334_subset.select(range(min(limit, len(no_conflict_6334_subset))))


