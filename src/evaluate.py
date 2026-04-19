from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from baseline_model import translate_baseline
from transformer_model import translate_transformer
from nltk.tokenize import word_tokenize

def tokenize(text):
    return word_tokenize(text.lower())



def evaluate_models(sample_size=None):
    dataset = load_dataset("parquet", data_files="../data/test/en-it_test.parquet")
    all_data = dataset["train"]
    if sample_size is None:
        eval_data = all_data
    else:
        if sample_size > len(all_data):
            raise ValueError("Sample size cannot be greater than dataset size")
        else:
            eval_data = all_data.shuffle(seed=100).select(range(sample_size))


    reference = []
    baseline_results = []
    transformer_results = []

    print("\n 5 Example Entries: ")
    for i, row in enumerate(eval_data):
        enu_ref = row["src_text"]
        ita_ref = row["tgt_text"]

        xform_out = translate_transformer(enu_ref)
        baseline_out = translate_baseline(enu_ref, 10)


        if i < 5:                                 ### print out a few results
            print("\nEnglish: ", enu_ref)
            print("Italian Reference: ", ita_ref)
            print("Baseline Translation: ", baseline_out)
            print("MarianMT Translation: ", xform_out)
            print()

        reference.append([tokenize(ita_ref)])   ### aaggghhhhhh
        baseline_results.append(tokenize(baseline_out))
        transformer_results.append(tokenize(xform_out))

    smooth=SmoothingFunction().method1

    bleu_baseline = corpus_bleu(reference, baseline_results, smoothing_function=smooth)
    bleu_xform = corpus_bleu(reference, transformer_results, smoothing_function=smooth)

    print("BLEU Baseline score: ", bleu_baseline)
    print("BLEU MarianMT score: ", bleu_xform)

if __name__ == "__main__":
    evaluate_models()