import collections

from sympy.printing.tests.test_tensorflow import tf

import helper
import numpy as np
import project_tests as tests
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Bidirectional, GRU, RepeatVector, TimeDistributed, Dense
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
import jieba
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
new_model = tf.keras.models.load_model('tr.h5')
new_model.summary()
english_sentences = helper.load_data2('data/small_vocab_ch3')
french_sentences = helper.load_data('data/small_vocab_ru3')

def tokenizeZh(x):
    text_sentences2=[]
    for words in x:
        seg_list = ' '.join(jieba.cut(words, cut_all=False))
        text_sentences2.append(seg_list)
    x_tk = Tokenizer(char_level=False)
    x_tk.fit_on_texts(text_sentences2)
    return x_tk.texts_to_sequences(text_sentences2), x_tk

def tokenize(x):

    x_tk = Tokenizer(char_level=False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post')
    tests.test_pad(pad)

def preprocess(x, y):
    preprocess_x, x_tk = tokenizeZh(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk


preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
    preprocess(english_sentences, french_sentences)

y_id_to_word = {value: key for key, value in french_tokenizer.word_index.items()}
y_id_to_word[0] = '<PAD>'
sentence = '他看到一辆黄色的旧卡车。'
sentence = [english_tokenizer.word_index[word] for word in jieba.cut(sentence, cut_all=False)]
sentence = pad_sequences([sentence], maxlen=preproc_english_sentences.shape[-1], padding='post')
sentences = np.array([sentence[0], preproc_english_sentences[0]])

predictions = new_model.predict(sentences, len(sentences))
print('Sample 1:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
print('Он увидел старый желтый грузовик.')
print('Sample 2:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
print(' '.join([y_id_to_word[np.max(x)] for x in preproc_french_sentences[0]]))