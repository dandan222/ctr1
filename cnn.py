import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import numpy as np

from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dense, Reshape, Lambda
from keras.optimizers import Adam
from scipy.fftpack import fft
from keras.models import Model
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import load_model
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# 获取信号的时频图
def compute_fbank(file):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, wavsignal = wav.read(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    window_length = fs / 1000 * time_window  # 计算窗长度的公式，目前全部为400固定值
    wav_arr = np.array(wavsignal)
    wav_length = len(wavsignal)
    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10  # 计算循环终止的位置，也就是最终生成的窗数
    # 	print(range0_end)
    data_input = np.zeros((range0_end, 200), dtype=np.float_)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float_)  # 窗口内的数据
    for i in range(0, range0_end):
        p_start = i * 160  # 步长10ms所以
        p_end = p_start + 400  # 窗口长25ms
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    # data_input = data_input[::]
    return data_input

#filepath = 'test.wav'
#a = compute_fbank(filepath)
# print(a.shape)
#plt.imshow(a.T, origin='lower')
#plt.show()
#a.shape
## 2. 数据处理

# #### 下载数据
# thchs30: http://www.openslr.org/18/

# ### 2.1 生成音频文件和标签文件列表
# 考虑神经网络训练过程中接收的输入输出。首先需要batch_size内数据需要统一数据的shape。
#
# **格式为**：[batch_size, time_step, feature_dim]
#
# 然而读取的每一个sample的时间轴长都不一样，所以需要对时间轴进行处理，选择batch内最长的那个时间为基准，进行padding。这样一个batch内的数据都相同，就能进行并行训练啦。

source_file = 'data_thchs30'
def source_get(source_file):
    train_file = source_file + '/data'
    label_lst = []
    wav_lst = []
    for root, dirs, files in os.walk(train_file):
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                wav_file = os.sep.join([root, file])
                label_file = wav_file + '.trn'
                wav_lst.append(wav_file)
                label_lst.append(label_file)

    return label_lst, wav_lst

label_lst, wav_lst = source_get(source_file)

print(label_lst[:10])
print(wav_lst[:10])

for i in range(10000):
    wavname = (wav_lst[i].split('/')[-1]).split('.')[0]
    labelname = (label_lst[i].split('/')[-1]).split('.')[0]
    if wavname != labelname:
        print('error')


def read_label(label_file):
    with open(label_file, 'r', encoding='utf8') as f:
        data = f.readlines()
        return data[1]


print(read_label(label_lst[0]))


def gen_label_data(label_lst):
    label_data = []
    for label_file in label_lst:
        pny = read_label(label_file)
        label_data.append(pny.strip('\n'))
    return label_data


label_data = gen_label_data(label_lst)
print(len(label_data))

def mk_vocab(label_data):
    vocab = []
    for line in label_data:
        line = line.split(' ')
        for pny in line:
            if pny not in vocab:
                vocab.append(pny)
    vocab.append('_')
    return vocab

vocab = mk_vocab(label_data)
# print(len(vocab))

def word2id(line, vocab):
    return [vocab.index(pny) for pny in line.split(' ')]

label_id = word2id(label_data[0], vocab)
print(label_data[0])
print(label_id)

print(vocab[:15])
print(label_data[10])
print(word2id(label_data[10], vocab))

fbank = compute_fbank(wav_lst[0])
print(wav_lst[0])
print(fbank.shape)
plt.imshow(fbank.T, origin = 'lower')
plt.show()
fbank = fbank[:fbank.shape[0]//8*8, :]
print(fbank.shape)

total_nums = 10000
batch_size = 4
batch_num = total_nums // batch_size
from random import shuffle
shuffle_list = [i for i in range(10000)]
shuffle(shuffle_list)

def get_batch(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(10000//batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            fbank = compute_fbank(wav_lst[index])
            fbank = fbank[:fbank.shape[0] // 8 * 8, :]
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(fbank)
            label_data_lst.append(label)
        yield wav_data_lst, label_data_lst

batch = get_batch(4, shuffle_list, wav_lst, label_data, vocab)
wav_data_lst, label_data_lst = next(batch)
for wav_data in wav_data_lst:
    print(wav_data.shape)
for label_data in label_data_lst:
    print(label_data)
lens = [len(wav) for wav in wav_data_lst]
print(max(lens))
print(lens)
def wav_padding(wav_data_lst):
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([leng//8 for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens

pad_wav_data_lst, wav_lens = wav_padding(wav_data_lst)
print(pad_wav_data_lst.shape)
print(wav_lens)
def label_padding(label_data_lst):
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
    for i in range(len(label_data_lst)):
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens

pad_label_data_lst, label_lens = label_padding(label_data_lst)
print(pad_label_data_lst.shape)
print(label_lens)

def data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(len(wav_lst)//batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            fbank = compute_fbank(wav_lst[index])
#             print(wav_lst[index])
            pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))
            pad_fbank[:fbank.shape[0], :] = fbank
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(pad_fbank)
            label_data_lst.append(label)
#             break
        pad_wav_data, input_length = wav_padding(wav_data_lst)
        pad_label_data, label_length = label_padding(label_data_lst)
        inputs = {'the_inputs': pad_wav_data,
                  'the_labels': pad_label_data,
                  'input_length': input_length,
                  'label_length': label_length,
                 }
#         print(pad_wav_data,pad_label_data,input_length,label_length)
        outputs = {'ctc': np.zeros(pad_wav_data.shape[0],)}
        yield inputs, outputs

def conv2d(size):
    return Conv2D(size, (3,3), use_bias=True, activation='relu',
        padding='same', kernel_initializer='he_normal')
def norm(x):
    return BatchNormalization(axis=-1)(x)
def maxpool(x):
    return MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(x)

def dense(units, activation="relu"):
    return Dense(units, activation=activation, use_bias=True,
        kernel_initializer='he_normal')
# x.shape=(none, none, none)
# output.shape = (1/2, 1/2, 1/2)
def cnn_cell(size, x, pool=True):
    x = norm(conv2d(size)(x))
    x = norm(conv2d(size)(x))
    if pool:
        x = maxpool(x)
    return x
def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class Amodel():
    """docstring for Amodel."""
    def __init__(self, vocab_size):
        super(Amodel, self).__init__()
        self.vocab_size = vocab_size
        self._model_init()
        self._ctc_init()
        self.opt_init()

    def _model_init(self):
        self.inputs = Input(name='the_inputs', shape=(None, 200, 1))
        self.h1 = cnn_cell(32, self.inputs)
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2)
        self.h4 = cnn_cell(128, self.h3, pool=False)
        # 200 / 8 * 128 = 3200
        self.h6 = Reshape((-1, 3200))(self.h4)
        self.h7 = dense(256)(self.h6)
        self.outputs = dense(self.vocab_size, activation='softmax')(self.h7)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)

    def _ctc_init(self):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')\
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs,
            self.input_length, self.label_length], outputs=self.loss_out)

    def opt_init(self):
        opt = Adam(lr = 0.0008, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)
        #self.ctc_model=multi_gpu_model(self.ctc_model,gpus=2)
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)
#     def load_model(self,file):
#         self.ctc_model.load_model(file)
#am = Amodel(1176)
#am.ctc_model.summary()
total_nums = 100
batch_size = 20
batch_num = total_nums // batch_size
epochs = 200


source_file = 'data_thchs30'
label_lst, wav_lst = source_get(source_file)
label_data = gen_label_data(label_lst[:100])
vocab = mk_vocab(label_data)
vocab_size = len(vocab)

# print(vocab_size)

shuffle_list = [i for i in range(100)]

am = Amodel(vocab_size)

for k in range(epochs):
    print('this is the', k+1, 'th epochs trainning !!!')
    #shuffle(shuffle_list)
    batch = data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab)
    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)
am.ctc_model.save_weights('res2.h5')
import json
with open('model_def.json','w') as ff:
    json_string = am.model.to_json()
    ff.write(json_string)#保存模型信息

sep = '\n'
fl=open('pinyin_list1.txt', 'w')
fl.write(sep.join(vocab))
fl.close()#保存拼音列表



