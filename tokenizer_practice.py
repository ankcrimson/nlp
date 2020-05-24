from keras.preprocessing.text import Tokenizer


sentences = [
    "i love my dog",
    "i love my cat",
    "you love my dog!"
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)

print(tokenizer.word_index)
