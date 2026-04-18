from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-it"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_transformer(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)

    return output

print(translate_transformer("my house"))
print(translate_transformer("I love you"))
print(translate_transformer("the car"))
print(translate_transformer("without you"))
print(translate_transformer("your table"))
print(translate_transformer("can I help you"))
print(translate_transformer("This evening I had dinner at a lovely restaurant before leaving for the airport."))

