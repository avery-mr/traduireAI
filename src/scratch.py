from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize

### OPUS BOOKS DATASET ###
# dataset = load_dataset("Helsinki-NLP/opus_books", "en-it")

### OPENSUBTITLES 2024 DATASET (LOCAL) $###
dataset = load_dataset(
    "parquet",
    data_files="../data/dev/en-it_dev.parquet"
)

print(dataset)
print(dataset["train"].column_names)
print(dataset["train"][0])