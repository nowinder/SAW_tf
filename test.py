# import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'

# 可能有报错matplotlib，由于不是专业版无scientific设置，加上代码
'''
import matplotlib
matplotlib.use('TKAgg)'
import matplotlib.pyplot as plt
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inputps import YtoZS, cacul
from net import IANN

weights_path_old = 'weights/model-ep152-valoss0.024'
weights_path_new = 'weights/model-ep228-valoss0.031'
freq = np.linspace(5*1e8,1.5*1e9,501)



musi = np.loadtxt('./datas/input/6pk_2.npymusi.csv')
mu = musi[:6]
sigma = musi[6:]

label_path = 'G:/Zheng_caizhi/Pycharmprojects/SAW_tf/datas/out/6p1k_2.csvmaxmin.csv'
label_mm = np.genfromtxt(label_path, delimiter=',')
label_max = label_mm[0]
label_min = label_mm[1]
label_path1 = 'G:/Zheng_caizhi/Pycharmprojects/SAW_tf/datas/out/6p1k_2.csv'
label = np.genfromtxt(label_path1, delimiter=',')

model_n = IANN()
model_o = IANN()
model_n.load_weights(weights_path_new)
model_o.load_weights(weights_path_old)

vali = True
test = False
if vali:
    data_path = 'G:/Zheng_caizhi/Pycharmprojects/SAW_tf/datas/input/vb.npy'
    data = np.load(data_path)
    test_set = tf.data.Dataset.from_tensor_slices(data).batch(128)
    test_result_n = model_n.predict(test_set)
    test_result_n = (test_result_n*(label_max - label_min)) + label_min
    test_result_o = model_o.predict(test_set)
    test_result_o = (test_result_o*(label_max - label_min)) + label_min

    plt.ion()
    for i in range(0,500):
        y_dl_n = cacul(test_result_n[i])
        y_dl_o = cacul(test_result_o[i])
        tt = cacul(label[i])
        plt.clf()
        # plt.ion()
        plt.plot(freq, abs(y_dl_n), label='NO_n')
        plt.plot(freq, abs(y_dl_o), label='NO_o')
        plt.plot(freq, abs(tt), label='GT')
        plt.yscale('log')
        plt.legend()

        plt.pause(2)
        if i==500:
            plt.ioff()
        plt.show(block=False)

        # plt.close()
# data = data * sigma + mu

if test:
    for i in range(19,101):
        test_set = np.loadtxt('./datas/h-0.01-0.01-1.csv',delimiter=',')
        tt = test_set[:, i:i+2]
        tt = np.vectorize(complex)(tt[:,0],tt[:,1])
        tt6 = np.array(YtoZS(tt,freq)).T
        test_su = (tt6 - mu) / sigma
        test_set = tf.data.Dataset.from_tensors(test_su).batch(1)
        test_result_n = model_n.predict(test_set)
# print(test_result_n.shape)
        test_result_n = (test_result_n[0]*(label_max - label_min)) + label_min
# print(test_result_n.shape)
# for i in range(0,101):
    # test_result_n = [4.02850237e+01,3.77054896e+02,1.45512831e+02,5.74451271e-02,2.11223737e+00,1.14177241e-03,4.20560322e+03]
    # y_dl = cacul(test_result_n[i,:])
        y_dl = cacul(test_result_n)
        # tt = cacul(label[i,:])
        plt.plot(freq,abs(y_dl),label='NO')
        plt.plot(freq,abs(tt),label='GT')
        plt.yscale('log')
        plt.legend()
        plt.show()

