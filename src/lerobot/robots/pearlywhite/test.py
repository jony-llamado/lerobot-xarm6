from datasets import load_dataset

ds = load_dataset("rdteteam/hello3")
for i, train in enumerate(ds["train"]):
    print(f'Train {i}: {train}\n')