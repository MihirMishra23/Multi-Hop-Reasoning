from datasets import Dataset

# Create train/eval splits
def create_train_val_splits(dataset: Dataset, train_size: int, eval_size: int, dataset_name: str):
    # TODO: how about SFT trainset split?

    dataset = dataset.shuffle(seed=42)  

    if dataset_name == "hotpotqa":
        # TODO: hack for now only for hotpotqa
        MAGIC_START_IDX = 82347
        MAGIC_TRAIN_MAX_SIZE = 8000
        MAGIC_VAL_MAX_SIZE   = 100

        assert MAGIC_START_IDX + MAGIC_TRAIN_MAX_SIZE + MAGIC_VAL_MAX_SIZE == len(dataset)
        assert train_size <= MAGIC_TRAIN_MAX_SIZE
        assert eval_size  <= MAGIC_VAL_MAX_SIZE

        n = len(dataset)
        assert train_size + MAGIC_VAL_MAX_SIZE <= n

        train_set = dataset.select(
            range(n - MAGIC_VAL_MAX_SIZE - train_size, n - MAGIC_VAL_MAX_SIZE)
        )
        eval_set = dataset.select(
            range(n - eval_size, n)
        )
    elif dataset_name == "mquake":
        assert train_size + eval_size <= len(dataset)

        train_set = dataset.select(range(0, train_size))
        eval_set = dataset.select(range(len(dataset)-eval_size, len(dataset)))
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return train_set, eval_set
