from datasets import load_dataset
import baseline_model, transformer_model


dataset = load_dataset(
    "parquet",
    data_files="../data/dev/en-it_dev.parquet"
)

print("dataset")
print(dataset)
print(dataset["test"][10]["src_text"])
# nltk.download("punkt")
# nltk.download("punkt_tab")