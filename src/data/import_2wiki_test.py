from data import get_dataset

ds = get_dataset("2wiki", "setting", "dev")
print(ds[0])