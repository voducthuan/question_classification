# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:54:25 2020

@author: Vo Duc Thuan
"""

import keras
import pandas as pd
import numpy as np
import matchzoo as mz
import json
import data
print('matchzoo version', mz.__version__)
print()

ranking_task = mz.tasks.Ranking()

print('data loading ...')
train_pack_raw = data.load_data('train', task='ranking')
test_pack_raw = data.load_data('test', task='ranking')
dev_pack_raw = data.load_data('dev', task='ranking')
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')

#Embbeded Glove
print("loading embedding ...")
glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=50)
print("embedding loaded as `glove_embedding`")

preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=40, remove_stop_words=False)
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

preprocessor.context
model = mz.models.ConvKNRM()
model.params.update(preprocessor.context)
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = glove_embedding.output_dim
model.params['embedding_trainable'] = True
model.params['filters'] = 128 
model.params['conv_activation_func'] = 'tanh' 
model.params['max_ngram'] = 3
model.params['use_crossmatch'] = True 
model.params['kernel_num'] = 11
model.params['sigma'] = 0.1
model.params['exact_sigma'] = 0.001
model.params['optimizer'] = 'adadelta'
model.build()
model.compile()

embedding_matrix = glove_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
model.load_embedding_matrix(embedding_matrix)

pred_x, pred_y = test_pack_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_y))

train_generator = mz.DataGenerator(
    train_pack_processed,
    mode='pair',
    num_dup=5,
    num_neg=1,
    batch_size=20
)
print('num batches:', len(train_generator))

history = model.fit_generator(train_generator, epochs=15, callbacks=[evaluate], workers=30, use_multiprocessing=True)




