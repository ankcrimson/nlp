from keras.layers import Embedding, GlobalAveragePooling1D, Dense
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json

datastore = []
with open('Sarcasm_Headlines_Dataset_v2.json', 'r') as jsonfile:
    datastore = jsonfile.readlines()

sentences = []
labels = []
urls = []
for line in datastore:
    item = json.loads(line)
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
print(len(sentences))

training_size = 20_000
vocabulary_size = 10000
max_length = 100
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


tokenizer = Tokenizer(num_words=vocabulary_size, oov_token='<OOV>')
tokenizer.fit_on_texts(training_sentences)

# print(tokenizer.word_index)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, maxlen=max_length, padding='post')

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, maxlen=max_length, padding='post')

print(training_padded[0])
print(training_padded.shape)

# Embedding is summing up the vectors in a multi dimensional space for word vectors towards sentiments

embedding_dim = 16


model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=max_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)
# testing

test_sentences = ["granny starting to fear spiders in the garden might be real",
                  "game of thrones season finale showing this sunday night"]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded_sequences = pad_sequences(
    test_sequences, maxlen=max_length, padding='post')
print(model.predict(test_padded_sequences))
