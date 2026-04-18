from datasets import load_dataset
from baseline_model import translate_baseline
from transformer_model import translate_transformer


test_set = load_dataset("parquet", data_files="../data/test/en-it_test.parquet")
#
# print("dataset")
# print(test_set)
# print
# ita = test_set["train"][100]["tgt_text"]
# print(ita)
# print(translate("how are you today?"))

for i in range(200, 230):
    enu = test_set["train"][i]["src_text"]
    ita = test_set["train"][i]["tgt_text"]
    print("English: ", enu)
    print("Target Italian: ", ita)
    print("Baseline Model:", translate_baseline(enu, 10))
    print("Transformer Model:", translate_transformer(enu))
    print("")