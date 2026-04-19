from datasets import load_dataset


dataset = load_dataset("parquet", data_files="../data/test/en-it_test.parquet")
eval_data = dataset["train"]
print(len(dataset))
