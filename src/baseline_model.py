from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from stopwords import english_stopwords, italian_stopwords, italian_skipwords

### OPUS BOOKS DATASET ###
# dataset = load_dataset("Helsinki-NLP/opus_books", "en-it")

### OPENSUBTITLES 2024 DATASET (LOCAL) $###
dataset = load_dataset(
    "parquet",
    data_files="../data/dev/en-it_dev.parquet"
)


# nltk.download("punkt")
# nltk.download("punkt_tab")

train_ds = dataset["train"].select(range(10000))



def tokenize(text):
    return word_tokenize(text.lower())        # keep everything lowercase for now

def is_word(token):
    return any(char.isalpha() for char in token)

### translation dictionary
en_to_it_counts = {}

for sentence in train_ds:
    en_text = sentence["src_text"]
    it_text = sentence["tgt_text"]

    en_tokens = tokenize(en_text)
    it_tokens = tokenize(it_text)

    ### remove punctuation for translation, getting in the way
    en_tokens = [t for t in en_tokens if is_word(t)]  #filter out punctuation and stop words
    it_tokens = [t for t in it_tokens if is_word(t)]

    for en_word in en_tokens:    ### run through each english word
        if en_word not in en_to_it_counts:
            en_to_it_counts[en_word] = {}  ### add english word to dictionary if necessary

        for it_word in it_tokens:
            if it_word not in en_to_it_counts[en_word]:
                en_to_it_counts[en_word][it_word] = 0
            en_to_it_counts[en_word][it_word] += 1

translation_probs = {}

for en_word, it_dict in en_to_it_counts.items():
    total = sum(it_dict.values())
    translation_probs[en_word] = {}

    for it_word, count in it_dict.items():
        translation_probs[en_word][it_word] = count / total

def best_translations(en_word, k=3):
    if en_word not in translation_probs:
        return []
    return sorted(translation_probs[en_word].items(), key=lambda x: x[1], reverse=True)[:k]

# print(best_translations("house"))
# print(best_translations("love"))
# print(best_translations("car"))
# print(best_translations("my"))
# print(best_translations("you"))
# print(best_translations("table"))
# print(best_translations("eat"))
# print(best_translations("without"))

### italian dictionaries
it_unigrams = {}
it_bigrams = {}

for sentence in train_ds:
    it_text = sentence["tgt_text"]
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

# print(sorted(it_unigrams.items(), key=lambda x: x[1], reverse=True)[:20])
# print(sorted(it_bigrams.items(), key=lambda x: x[1], reverse=True)[:20])


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
    word_1_count = it_unigrams.get(word1, 0)

    if word_1_count == 0:      # just in case
        return 0

    return word_1_2_count / word_1_count

# print(bigram_prob(',', 'che'))
# print(bigram_prob('!', '</s>'))
# print(bigram_prob('e', 'mi'))
# print(bigram_prob('con', 'le'))
# print(bigram_prob('la', 'mia'))
# print(bigram_prob(',', 'di'))
# print(bigram_prob('di', ','))


#### putting it all together
#### score = translation_prob(e_i, w_i) * bigram_prob(prev_word, w_i)

def translate(en_sentence, k=3):
    en_tokens = [t for t in tokenize(en_sentence) if is_word(t)]

    output = []
    prev_word = "<s>"    ### include start of sentence for new sentence, since there isn't a previous word

    for en_word in en_tokens:
        word_options = best_translations(en_word, k)

        if not word_options:
            continue

        best_word = None
        best_score = -1     # start with a negative score

        for it_word, trans_prob in word_options:
            order_prob = max(bigram_prob(prev_word, it_word), 1e-6)  # just in case order_prob = 0
            # score = order_prob * (trans_prob ** 2)
            score = trans_prob + 0.1 * order_prob
            # print("score:" , score)

            if score > best_score:
                best_score = score
                best_word = it_word

        if best_word is not None:
            output.append(best_word)
            prev_word = best_word

    return " ".join(output)








if __name__ == "__main__":
    # print(translate("my house"))
    # print(translate("I love you"))
    # print(translate("the car"))
    # print(translate("without you"))
    # print(translate("your table"))
    # print(translate("can I help you"))
    print(translate("yesterday I went to the market with my sister"))
    print(translate("how are you today?"))
















# @inproceedings{tiedemann-luo-2026-opensubtitles2024,
#   title={OpenSubtitles2024: A Massively Parallel Dataset of Movie Subtitles for MT Development and Evaluation},
#   author={Tiedemann, Jörg and Luo, Hengyu},
#   booktitle={Proceedings of the 15th edition of the Language Resources and Evaluation Conference (LREC 2026)},
#   year={2026}
# }