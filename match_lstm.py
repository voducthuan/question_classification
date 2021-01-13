# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:57:52 2020

@author: Vo Duc Thuan
"""

import keras
import pandas as pd
import numpy as np
import matchzoo as mz
import json
import data

print(mz.__version__)
task = mz.tasks.Ranking() # Ranking task
print(task)

#Load data
print('data loading ...')
train_pack_raw = data.load_data('train', task='ranking')
test_pack_raw = data.load_data('test', task='ranking')
dev_pack_raw = data.load_data('dev', task='ranking')
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')


#Load GloVe data
print("loading embedding ...")
glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
print("embedding loaded as `glove_embedding`")


preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=40, remove_stop_words=False)

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

preprocessor.context

# Set up model
model = mz.contrib.models.MatchLSTM()
model.params.update(preprocessor.context)
model.params['task'] = task
model.params['embedding_output_dim'] = 100
model.params['embedding_trainable'] = True
model.params['fc_num_units'] = 100
model.params['lstm_num_units'] = 100
model.params['dropout_rate'] = 0.5
model.params['optimizer'] = 'adadelta'
model.guess_and_fill_missing_params()

print(model.params.completed())

model.build()
model.compile()
print(model.params)
model.backend.summary()

# Build embedding matrix (GloVe)
embedding_matrix = glove_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
model.load_embedding_matrix(embedding_matrix)

# Drop label (no label) for test_x, test_y for evaluation in the model
test_x, test_y = test_pack_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=test_x, y=test_y, batch_size=len(test_x))

# Devide training datapack into batches
train_generator = mz.DataGenerator(
    train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1,
    batch_size=20
)
print('num batches:', len(train_generator))

# Train model with epochs=15
history = model.fit_generator(train_generator, epochs=15, callbacks=[evaluate], workers=4, use_multiprocessing=True)