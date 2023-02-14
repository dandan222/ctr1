import collections
import keras
from keras.preprocessing.text import Tokenizer

import helper


english_sentences = helper.load_data('data/small_vocab_en')
french_sentences = helper.load_data('data/small_vocab_fr')
print('Dataset Loaded')

for sample_i in range(2):
 print('small_vocab_en Line {}: {}'.format(sample_i + 1, english_sentences[sample_i]))
 print('small_vocab_fr Line {}: {}'.format(sample_i + 1, french_sentences[sample_i]))

english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])
print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')

def tokenize(x):
 x_tk = Tokenizer(char_level = False)
 x_tk.fit_on_texts(x)
 return x_tk.texts_to_sequences(x), x_tk
text_sentences = [
 'The quick brown fox jumps over the lazy dog .',
 'By Jove , my quick study of lexicography won a prize .',
 'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
 print('Sequence {} in x'.format(sample_i + 1))
 print(' Input: {}'.format(sent))
 print(' Output: {}'.format(token_sent))

def pad(x, length=None):
 if length is None:
 length = max([len(sentence) for sentence in x])
 return pad_sequences(x, maxlen = length, padding = 'post')
tests.test_pad(pad)
# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
 print('Sequence {} in x'.format(sample_i + 1))
 print(' Input: {}'.format(np.array(token_sent)))
 print(' Output: {}'.format(pad_sent))

def preprocess(x, y):
 preprocess_x, x_tk = tokenize(x)
 preprocess_y, y_tk = tokenize(y)
preprocess_x = pad(preprocess_x)
 preprocess_y = pad(preprocess_y)
# Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
 preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
return preprocess_x, preprocess_y, x_tk, y_tk
preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
 preprocess(english_sentences, french_sentences)

max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)
print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

def logits_to_text(logits, tokenizer):
 index_to_words = {id: word for word, id in tokenizer.word_index.items()}
 index_to_words[0] = '<PAD>'
return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
print('`logits_to_text` function loaded.')



def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):

 model = Sequential()
 model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
 model.add(Bidirectional(GRU(256,return_sequences=False)))
 model.add(RepeatVector(output_sequence_length))
 model.add(Bidirectional(GRU(256,return_sequences=True)))
 model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
 learning_rate = 0.005

 model.compile(loss = sparse_categorical_crossentropy,
 optimizer = Adam(learning_rate),
 metrics = ['accuracy'])

 return model
tests.test_model_final(model_final)
print('Final Model Loaded')
