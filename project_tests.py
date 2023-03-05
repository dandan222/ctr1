import numpy as np
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


def _test_model(model, input_shape, output_sequence_length, french_vocab_size):
    if isinstance(model, Sequential):
        model = model.model

    assert model.input_shape == (None, *input_shape[1:]),\
        'Wrong input shape. Found input shape {} using parameter input_shape={}'.format(model.input_shape, input_shape)

    assert model.output_shape == (None, output_sequence_length, french_vocab_size),\
        'Wrong output shape. Found output shape {} using parameters output_sequence_length={} and french_vocab_size={}'\
            .format(model.output_shape, output_sequence_length, french_vocab_size)

    assert len(model.loss_functions) > 0,\
        'No loss function set.  Apply the `compile` function to the model.'

    assert sparse_categorical_crossentropy in model.loss_functions,\
        'Not using `sparse_categorical_crossentropy` function for loss.'


def test_tokenize(tokenizeZh):
    sentences = [
        '敏捷的棕色狐狸跳过懒惰的狗.',
        '天啊，我对词汇学的快速研究赢得了一个奖项 .',
        '这是一个简短的句子 .']
    tokenized_sentences, tokenizer = tokenizeZh(sentences)
    assert tokenized_sentences == tokenizer.texts_to_sequences(sentences),\
        'Tokenizer returned and doesn\'t generate the same sentences as the tokenized sentences returned. '


def test_pad(pad):
    tokens = [
        [i for i in range(4)],
        [i for i in range(6)],
        [i for i in range(3)]]
    padded_tokens = pad(tokens)
    padding_id = padded_tokens[0][-1]
    true_padded_tokens = np.array([
        [i for i in range(4)] + [padding_id]*2,
        [i for i in range(6)],
        [i for i in range(3)] + [padding_id]*3])
    assert isinstance(padded_tokens, np.ndarray),\
        'Pad returned the wrong type.  Found {} type, expected numpy array type.'
    assert np.all(padded_tokens == true_padded_tokens), 'Pad returned the wrong results.'

    padded_tokens_using_length = pad(tokens, 9)
    assert np.all(padded_tokens_using_length == np.concatenate((true_padded_tokens, np.full((3, 3), padding_id)), axis=1)),\
        'Using length argument return incorrect results'


def test_model_final(model_final):
    input_shape = (286, 33)
    output_sequence_length = 28
    english_vocab_size = 283
    french_vocab_size = 415

    model = model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)
