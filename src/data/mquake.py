from datasets import load_dataset, concatenate_datasets

def _rename_question(example):
    example["question"] = example["questions"]
    example["answers"] = [example["answer"]] + example["answer_alias"]
    return example


def _filter_example(example):
    labels = example['split']['ALL']
    return 'train_edited' in labels or 'test_edited' in labels 


def load_mquake(split : str, limit : int,  seed : int):
    raw_dataset = load_dataset("henryzhongsc/MQuAKE-Remastered", split = "CF6334") #Fixed constant split

    no_conflict_6334_subset  = raw_dataset.filter(_filter_example)
    # See code examples here for selecting the dataset: https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered
    # This selects the 6334 subset which we can freely edit without contamination.

    if limit is None:
        limit = len(no_conflict_6334_subset)
        
    if split == 'train':
        no_conflict_6334_subset = no_conflict_6334_subset.select(range(5334))
    
    if split == 'test':
        no_conflict_6334_subset.select(range(5334, 6334))
    no_conflict_6334_subset = no_conflict_6334_subset.map(_rename_question, load_from_cache_file=False)
    return no_conflict_6334_subset
