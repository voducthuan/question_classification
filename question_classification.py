# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 04:20:05 2020

@author: Vo Duc Thuan
"""

import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

vocab_size = 3000 # size of vocabulary
embedding_dim = 64
max_length = 20
training_portion = .80 # set ratio of train (80%) and validation (20%)

list_of_questions = []
labels = []

# Read data and remove stopword
with open("data/train_5500.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        question = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            question = question.replace(token, ' ')
            question = question.replace(' ', ' ')
        list_of_questions.append(question)
print(len(labels))
print(len(list_of_questions))

train_size = int(len(list_of_questions) * training_portion)
train_questions = list_of_questions[0: train_size]
train_labels = labels[0: train_size]
validation_questions = list_of_questions[train_size:]
validation_labels = labels[train_size:]

tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(train_questions)
word_index = tokenizer.word_index

dict(list(word_index.items())[0:100]) ## print out first 100 index of vocabulary

train_sequences = tokenizer.texts_to_sequences(train_questions)

# First of 50 records in token form
for i in range(50):
    print(train_sequences[i])

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')

# First of 50 records after padding to size 20
for i in range(50):
    print(train_padded[i])

validation_sequences = tokenizer.texts_to_sequences(validation_questions)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding='post', truncating='post')

# set of lables
print(set(labels))

# label to token
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

# First of 50 labels (token form)
for i in range(50):
    print(training_label_seq[i])

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Checking encode and original
def decode_question(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print('------------------------')
print(decode_question(train_padded[20]))
print(train_questions[20])
print('------------------------')

# Use tf.keras.layers.Bidirectional(tf.keras.layers.LSTM()).
# Use ReLU in place of tanh function.
# Add a Dense layer with 7 units and softmax activation.

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Traing model with 15 epochs
num_epochs = 15
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

# Predict input text
question_input = ["What metal has the highest melting point ?"]
seq = tokenizer.texts_to_sequences(question_input)
padded = pad_sequences(seq, maxlen=max_length)
prediction = model.predict(padded)
print(prediction)
print(labels[np.argmax(prediction)])