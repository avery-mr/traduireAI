from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from baseline_model import translate_baseline
from transformer_model import translate_transformer
from nltk.tokenize import word_tokenize

def tokenize(text):
    return word_tokenize(text.lower())



def evaluate_all():
    dataset = load_dataset("parquet", data_files="../data/test/en-it_test.parquet")

    # eval_data = dataset["train"].select(range(100))
    eval_data = dataset["train"]

    reference = []
    baseline_results = []
    transformer_results = []

    for i, row in enumerate(eval_data):
        enu_ref = row["src_text"]
        ita_ref = row["tgt_text"]

        xform_out = translate_transformer(enu_ref)
        baseline_out = translate_baseline(enu_ref, 10)

        # if i < 10:                                 ### print out a few results
        #     print("English: ", enu_ref)
        #     print("Italian Reference: ", ita_ref)
        #     print("Baseline Translation: ", baseline_out)
        #     print("MarianMT Translation: ", xform_out)
        #     print()

        reference.append([tokenize(ita_ref)])   ### aaggghhhhhh
        baseline_results.append(tokenize(baseline_out))
        transformer_results.append(tokenize(xform_out))

    smooth=SmoothingFunction().method1

    bleu_baseline = corpus_bleu(reference, baseline_results, smoothing_function=smooth)
    bleu_xform = corpus_bleu(reference, transformer_results, smoothing_function=smooth)

    print("BLEU Baseline score: ", bleu_baseline)
    print("BLEU MarianMT score: ", bleu_xform)

if __name__ == "__main__":
    evaluate_all()