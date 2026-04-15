from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize

validation_ds = load_dataset(
    "Helsinki-NLP/OpenSubtitles2024",
    split="validation",
    trust_remote_code=True,
    language_pairs="en-it",
)
# dataset = load_dataset("Helsinki-NLP/opus_books", "en-it")
# print(dataset)

# nltk.download("punkt")
# nltk.download("punkt_tab")

# train_ds = dataset["train"].select(range(10000))



def tokenize(text):
    return word_tokenize(text.lower())        # keep everything lowercase for now


# little_test = dataset["train"][100]
little_test = dataset["train"].select(range(10000))
# print(little_test['translation']['it'])

# en_text = little_test["translation"]["en"]
# it_text = little_test["translation"]["it"]



### italian dictionaries
it_unigrams = {}
it_bigrams = {}

for sentence in little_test:
    it_text = sentence["translation"]["it"]
    # print(it_test_text)
    it_tokens = ["<s>"] + tokenize(it_text) + ["</s>"]  # add sentence start and stop 'characters', helps learn how sentence is structured
    # print(it_test_tokens)

    for token in it_tokens:
        if token not in it_unigrams:
            it_unigrams[token] = 0
        it_unigrams[token] += 1

    # print(it_unigrams)

    for i in range(len(it_tokens) - 1):
        token_pair = (it_tokens[i], it_tokens[i+1])
        # print(token_pair)

        if token_pair not in it_bigrams:
            it_bigrams[token_pair] = 0
        it_bigrams[token_pair] += 1

print(sorted(it_unigrams.items(), key=lambda x: x[1], reverse=True)[:20])
print(sorted(it_bigrams.items(), key=lambda x: x[1], reverse=True)[:20])


#### --- bigram probabilities ----
#### P(word2 | word1) = occurences of (word1, word2) / occurences of (word1)
#### P("voglio" | "ti") = count("ti", "voglio") / count("ti")
###

# word_1_2_count = it_bigrams.get(("non", "mi"), 0)
# word_1_count = it_unigrams.get("non", 0)
# prob = word_1_2_count / word_1_count
# print(prob)

def bigram_prob(word1, word2):
    word_1_2_count = it_bigrams.get((word1, word2), 0)
    word_1_count = it_unigrams.get(word1)
    return(word_1_2_count / word_1_count)

print(bigram_prob(',', 'che'))
print(bigram_prob('!', '</s>'))
print(bigram_prob('e', 'mi'))
print(bigram_prob('con', 'le'))
print(bigram_prob('la', 'mia'))
print(bigram_prob(',', 'di'))
print(bigram_prob('di', ','))


























# @inproceedings{tiedemann-luo-2026-opensubtitles2024,
#   title={OpenSubtitles2024: A Massively Parallel Dataset of Movie Subtitles for MT Development and Evaluation},
#   author={Tiedemann, Jörg and Luo, Hengyu},
#   booktitle={Proceedings of the 15th edition of the Language Resources and Evaluation Conference (LREC 2026)},
#   year={2026}
# }