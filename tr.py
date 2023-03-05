import collections
import helper
import numpy as np
import project_tests as tests
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Bidirectional, GRU, RepeatVector, TimeDistributed, Dense
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam, TFOptimizer
from keras.models import Model
from keras.models import Sequential
import jieba
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

english_sentences = helper.load_data2('data/small_vocab_ch3')
french_sentences = helper.load_data('data/small_vocab_ru3')
print('Dataset Loaded')

for sample_i in range(2):
    print('small_vocab_ch Line {}: {}'.format(sample_i + 1, english_sentences[sample_i]))
    print('small_vocab_ru Line {}: {}'.format(sample_i + 1, french_sentences[sample_i]))

english_words_counter = collections.Counter([word for sentence in english_sentences  for word in jieba.cut(sentence, cut_all=False)])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])
print('{} chinese words.'.format(len([word for sentence in english_sentences for word in jieba.cut(sentence, cut_all=False)])))
print('{} unique chinese words.'.format(len(english_words_counter)))
print('10 Most common words in the chinese dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} russia words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique russia words.'.format(len(french_words_counter)))
print('10 Most common words in the russia dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')
# 对于中文
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

text_sentences = [
    '敏捷的棕色狐狸跳过懒惰的狗。',
    '天啊，我对词汇学的快速研究赢得了一个奖项。',
    '这是一个简短的句子。']
text_tokenized, text_tokenizer = tokenizeZh(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print(' Input: {}'.format(sent))
    print(' Output: {}'.format(token_sent))


def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post')
    tests.test_pad(pad)
    # Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print(' Input: {}'.format(np.array(token_sent)))
    print(' Output: {}'.format(pad_sent))


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



max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)
print('Data Preprocessed')
print("Max chinese sentence length:", max_english_sequence_length)
print("Max russia sentence length:", max_french_sequence_length)
print("chinese vocabulary size:", english_vocab_size)
print("russia vocabulary size:", french_vocab_size)


def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


print('`logits_to_text` function loaded.')


def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size, output_dim=128, input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    learning_rate = 0.005

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])

    return model


# tests.test_model_final(model_final)
print('Final Model Loaded')


def final_predictions(x, y, x_tk, y_tk):
    tmp_X = pad(preproc_english_sentences)
    model = model_final(tmp_X.shape,
                        preproc_french_sentences.shape[1],
                        len(english_tokenizer.word_index) + 1,
                        len(french_tokenizer.word_index) + 1)

    model.fit(tmp_X, preproc_french_sentences, batch_size=1024, epochs=200, validation_split=0.2)
    model.save('tr.h5')

    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'
    sentence = '他看到一辆黄色的旧卡车。'
    sentence = [x_tk.word_index[word] for word in jieba.cut(sentence, cut_all=False)]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])

    predictions = model.predict(sentences, len(sentences))
    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Он увидел старый желтый грузовик.')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))

final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)
