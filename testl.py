import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import keras
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
py = []
f = open('pinyin_list.txt')
contents = f.readlines()
for i in contents:
    i = i.strip('\n')
    py.append(i)


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


def wav_padding(wav_data_lst):
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([leng // 8 for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens


def decode_ctc(num_result, num2word):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype=np.int32)
    in_len[0] = result.shape[1];
    r = K.ctc_decode(result, in_len, greedy=True, beam_width=10, top_paths=1)
    r1 = K.get_value(r[0][0])
    r1 = r1[0]
    text = []
    for i in r1:
        text.append(num2word[i])
    return r1, text
wav_data_lst=[]
fbank = compute_fbank('test.wav')
fbank = fbank[:fbank.shape[0] // 8 * 8, :]
wav_data_lst.append(fbank)
inputdata,outher = wav_padding(wav_data_lst)

with open('model_def.json') as ff:
    model_json=ff.read()
    model=keras.models.model_from_json(model_json)
model.load_weights('res2.h5')

preds=model.predict(inputdata)
result, text = decode_ctc(preds, py)

print(result,text)
from LanguageModel2 import ModelLanguage
ml = ModelLanguage('model_language')
ml.LoadModel()
str_pinyin = text
r = ml.SpeechToText(str_pinyin)
print('语音转文字结果：\n',r)

