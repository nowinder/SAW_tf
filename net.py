# %load net_2paras
# code by Zhengcaizhi
import os
import random

import keras.optimizers as optimizer
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Input, Model, backend
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
# import skrf as rf
# from data_gene_2paras import cacul
from keras.layers import Lambda, Concatenate, Dense, Conv1D, Flatten, MaxPool1D


def IANN():
    input = Input(shape=(1001, 6))
    branch1 = Lambda(lambda x: x[:, :, 0])(input)
    branch2 = Lambda(lambda x: x[:, :, 1])(input)
    branch3 = Lambda(lambda x: x[:, :, 2])(input)
    branch4 = Lambda(lambda x: x[:, :, 3])(input)
    branch5 = Lambda(lambda x: x[:, :, 4])(input)
    branch6 = Lambda(lambda x: x[:, :, 5])(input)
    h1 = Dense(2003, activation='relu', kernel_initializer='he_uniform')(branch1)
    h2 = Dense(2003, activation='relu', kernel_initializer='he_uniform')(branch2)
    h3 = Dense(2003, activation='relu', kernel_initializer='he_uniform')(branch3)
    h4 = Dense(2003, activation='relu', kernel_initializer='he_uniform')(branch4)
    h5 = Dense(2003, activation='relu', kernel_initializer='he_uniform')(branch5)
    h6 = Dense(2003, activation='relu', kernel_initializer='he_uniform')(branch6)
    h1 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h1)
    # h1 = Dropout(0.2)(h1)
    h2 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h2)
    # h2 = Dropout(0.2)(h2)
    h3 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h3)
    # h3 = Dropout(0.2)(h3)
    h4 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h4)
    # h4 = Dropout(0.2)(h4)
    h5 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h5)
    # h5 = Dropout(0.2)(h5)
    h6 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h6)
    # h6 = Dropout(0.2)(h6)
    h1 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h1)
    # h1 = Dropout(0.2)(h1)
    h2 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h2)
    # h2 = Dropout(0.2)(h2)
    h3 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h3)
    # h3 = Dropout(0.2)(h3)
    h4 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h4)
    # h4 = Dropout(0.2)(h4)
    h5 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h5)
    # h5 = Dropout(0.2)(h5)
    h6 = Dense(1001, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h6)
    # h6 = Dropout(0.2)(h6)

    # h1 = Dense(3207,activation='relu', kernel_initializer='he_uniform')(h1)
    # h2 = Dense(3207,activation='relu', kernel_initializer='he_uniform')(h2)
    # h3 = Dense(3207,activation='relu', kernel_initializer='he_uniform')(h3)
    # h4 = Dense(3207,activation='relu', kernel_initializer='he_uniform')(h4)
    # h5 = Dense(3207,activation='relu', kernel_initializer='he_uniform')(h5)
    # hn drop?
    # o1 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h1)
    # o2 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h2)
    # o3 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h3)
    # o4 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h4)
    # o5 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h5)
    # o6 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h6)
    # out1 = Concatenate(axis = -1)((o1,o2,o3,o4,o5,o6))
    out1 = Concatenate(axis=-1)((h1, h2, h3, h4, h5, h6))
    # out1 = Dense(100, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(out1) #model2v6
    # out1 = Dense(100, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(out1) #model2v6
    # out1 = Dense(2045, activation='relu', kernel_initializer='he_uniform',
    #              kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    out1 = Dense(1023, activation='relu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    out1 = Dense(511, activation='relu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    # out1 = Dropout(0.2)(out1)
    out1 = Dense(255, activation='relu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    # out1 = Dropout(0.2)(out1)
    out1 = Dense(127, activation='relu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    # out1 = Dropout(0.2)(out1)
    out1 = Dense(63, activation='relu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    # out1 = Dropout(0.2)(out1)
    out1 = Dense(31, activation='relu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    # out1 = Dropout(0.2)(out1)
    h2 = Dense(15, activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)
    out2 = Dense(7, activation='sigmoid', kernel_initializer='he_uniform')(h2)
    model = Model(inputs=input, outputs=out2)
    model.summary()
    return model

def IANN2():
    input = Input(shape=(1001, 2))
    branch1 = Lambda(lambda x: x[:, :, 0])(input)
    branch2 = Lambda(lambda x: x[:, :, 1])(input)
    h1 = Dense(2003, activation='gelu', kernel_initializer='he_uniform')(branch1)
    h2 = Dense(2003, activation='gelu', kernel_initializer='he_uniform')(branch2)
    h1 = Dense(4007, activation='gelu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h1)
    # h1 = Dropout(0.2)(h1)
    h2 = Dense(4007, activation='gelu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(h2)
    # h2 = Dropout(0.2)(h2)
    # h6 = Dropout(0.2)(h6)

    # h1 = Dense(3207,activation='relu', kernel_initializer='he_uniform')(h1)
    # h2 = Dense(3207,activation='relu', kernel_initializer='he_uniform')(h2)
    # h3 = Dense(3207,activation='relu', kernel_initializer='he_uniform')(h3)
    # h4 = Dense(3207,activation='relu', kernel_initializer='he_uniform')(h4)
    # h5 = Dense(3207,activation='relu', kernel_initializer='he_uniform')(h5)
    # hn drop?
    # o1 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h1)
    # o2 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h2)
    # o3 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h3)
    # o4 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h4)
    # o5 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h5)
    # o6 = Dense(16,activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(h6)
    # out1 = Concatenate(axis = -1)((o1,o2,o3,o4,o5,o6))
    out1 = Concatenate(axis=-1)((h1, h2))
    # out1 = Dense(100, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(out1) #model2v6
    # out1 = Dense(100, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00,l2=0.00000))(out1) #model2v6
    out1 = Dense(2045, activation='gelu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    out1 = Dense(1023, activation='gelu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    out1 = Dense(511, activation='gelu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    # out1 = Dropout(0.2)(out1)
    out1 = Dense(255, activation='gelu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    # out1 = Dropout(0.2)(out1)
    out1 = Dense(127, activation='gelu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    # out1 = Dropout(0.2)(out1)
    out1 = Dense(63, activation='gelu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    # out1 = Dropout(0.2)(out1)
    out1 = Dense(31, activation='gelu', kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)  # model2v6
    # out1 = Dropout(0.2)(out1)
    h2 = Dense(15, activation='gelu', kernel_initializer='he_uniform',
               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.00000))(out1)
    # h2 = MaxPool1D()(h2)
    out2 = Dense(7, activation='sigmoid', kernel_initializer='he_uniform')(h2)
    model = Model(inputs=input, outputs=out2)
    model.summary()
    return model

def conv():  # model3v1
    input = Input(shape=(801, 5))
    branchs = Lambda(lambda x: tf.split(x, num_or_size_splits=5, axis=-1))(input)
    o1 = Conv1D(4, 256, activation='relu', kernel_initializer='he_uniform')(branchs[0])
    o2 = Conv1D(4, 256, activation='relu', kernel_initializer='he_uniform')(branchs[1])
    o3 = Conv1D(4, 256, activation='relu', kernel_initializer='he_uniform')(branchs[2])
    o4 = Conv1D(4, 256, activation='relu', kernel_initializer='he_uniform')(branchs[3])
    o5 = Conv1D(4, 256, activation='relu', kernel_initializer='he_uniform')(branchs[4])
    o1 = Conv1D(16, 256, activation='relu', kernel_initializer='he_uniform')(o1)
    o2 = Conv1D(16, 256, activation='relu', kernel_initializer='he_uniform')(o2)
    o3 = Conv1D(16, 256, activation='relu', kernel_initializer='he_uniform')(o3)
    o4 = Conv1D(16, 256, activation='relu', kernel_initializer='he_uniform')(o4)
    o5 = Conv1D(16, 256, activation='relu', kernel_initializer='he_uniform')(o5)
    out1 = Concatenate(axis=-1)((o1, o2, o3, o4, o5))
    c1 = Conv1D(64, 256, activation='relu', kernel_initializer='he_uniform')(out1)
    out1 = Flatten()(c1)
    out2 = Dense(2, activation='sigmoid', kernel_initializer='he_uniform')(out1)
    model = Model(inputs=input, outputs=out2)
    model.summary()
    return model


# names = [[150,45,80],[150,50,80],[150,55,80]
#          ,[60,45,80],[60,50,80],[60,55,80]
#          ,[90,45,80],[90,50,80],[90,55,80]]

# MP_num = [45,60,65,55,50]
# MP_num = [45]
# Pitch_num = [180,200,225,260]
# PF = [[180,8.5],[200,7.5],[225,6.5],[260,5.5]]
# PF = [[200,7.5]]

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=800,
    #   10的decay训练有些过于慢了
    decay_rate=1,
    staircase=False)

def scheduler(epoch):
    # lr = backend.get_value(model.optimizer.lr)
    # if epoch < 1000:
    #     backend.set_value(model.optimizer.lr,0.001)
    
    if epoch >=200:
        new_lr = lr_schedule(epoch)
        backend.set_value(model.optimizer.lr, new_lr)
    return backend.get_value(model.optimizer.lr)

if __name__ == '__main__':
    model = IANN2()
    # freq = np.linspace(5 * 1e8, 1.5 * 1e9, 501)
    data_path = 'G:/Zheng_caizhi/Pycharmprojects/SAW_tf/datas/input/7p9w_log.npy'
    data = np.load(data_path)
    vali_path = 'G:/Zheng_caizhi/Pycharmprojects/SAW_tf/datas/input/vali_logn.npy'
    vali = np.load(vali_path)
    # data2_path = 'D:\\data\\7p/input2w/2w_0.5.npy'
    # data2 = np.load(data2_path)
    # data = np.concatenate((data1, data2))
    label_path1 = 'G:/Zheng_caizhi/Pycharmprojects/SAW_tf/datas/out/7p9w.csv'
    # label_path2 = 'D:\\data\\7p/out/MP60_0.5.csv'
    label = np.genfromtxt(label_path1, delimiter=',')
    valilabel_path1 = 'G:/Zheng_caizhi/Pycharmprojects/SAW_tf/datas/out/vali600.csv'
    valilabel = np.genfromtxt(valilabel_path1, delimiter=',')
    # label2 = np.genfromtxt(label_path2, delimiter=',')
    # label = np.concatenate((label1, label2))
    label_mm_path = label_path1 + 'maxmin.csv'
    label_mm = np.genfromtxt(label_mm_path, delimiter=',')
    label_max = label_mm[0]
    label_min = label_mm[1]
    label = (label - label_min) / (label_max - label_min)
    valilabel = (valilabel - label_min) / (label_max - label_min)
    data_set = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(data).map(lambda x:x[:,:2]), tf.data.Dataset.from_tensor_slices(label)))
    vali_set = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(vali).map(lambda x:x[:,:2]), tf.data.Dataset.from_tensor_slices(valilabel)))
    BATCHSIZE = 128
    # dataset = data_set
    # valiset = dataset.take(128).batch(BATCHSIZE).cache().prefetch(tf.data.AUTOTUNE)
    # trainset = dataset.skip(128)
    se =random.randint(1,int(10e2))
    print('seed=%d' % se) # 333
    # se=333
    trainset = data_set.cache().shuffle(data_set.cardinality(), seed=se, reshuffle_each_iteration=True).repeat().batch(
        BATCHSIZE).prefetch(tf.data.AUTOTUNE)
    valiset = vali_set.batch(BATCHSIZE).cache().prefetch(tf.data.AUTOTUNE)

    checkpoint_filepath = os.path.join(os.getcwd(), 'weights', 'model-ep{epoch:03d}-valoss{val_loss:.3f}')
    # shutil.rmtree(checkpoint_filepath)
    # checkpoint_filepath = checkpoint_filepath + '\\' +str(Pitch)
    # log_dir="c:\\Users\\caizhi.zheng\\code\\For AI\\logs2v5\\"+str(MP)+'-'+str(Pitch)
    log_dir = os.path.join(os.getcwd(), 'logs')
    # shutil.rmtree(log_dir,ignore_errors=True)

    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='loss', verbose=1, save_weights_only=True,save_best_only=True, initial_value_threshold=0.003)
    early_stop = EarlyStopping(monitor='val_loss', patience=500)
    Board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=20)#, profile_batch='10, 20')
    # opt = optimizer.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer.Adam(learning_rate=0.001), loss='mse', metrics='accuracy')
    reduce_lr = LearningRateScheduler(scheduler)
    # model.reset_states()
    # model.reset_metrics()
    # model.load_weights(checkpoint_filepath)
    # reduce_lr = LearningRateScheduler(scheduler)
    # model.fit(dataset, epochs=25)
    history = model.fit(trainset, validation_data=valiset, steps_per_epoch=80, epochs=5000,
                        callbacks=[model_checkpoint, reduce_lr, early_stop, Board])  # validation_split=0.02,)
    model.evaluate(valiset)
    print('seed=%d' % se)
    # test
    # if Pitch != 225:
    #     pitch = int(0.1*Pitch)
    # test_path = 'C:\\Users\\caizhi.zheng\\code\\For AI\\SNP Selection/'+'LT'+str(NT)+ 'MP'+str(MP) +'P'+str(pitch) +'/'
    # saw_set = rf.read_all_networks(test_path)
    # for name in saw_set:
    #     saw = saw_set[name]
    # y0 = saw['900-1100MHZ'].y[:,0,0]
    # test_set = np.ones((801,5))*0.01
    # # test_set = np.zeros((801,5))
    # # data_mm_path = data_path + 'musi.csv'
    # # data_mm = np.genfromtxt(data_mm_path, delimiter=',')
    # # data_mu = data_mm[:5]
    # # data_sigma = data_mm[5:10]
    # # test_set = (test_set * data_sigma) + data_mu

    # test_set = tf.data.Dataset.from_tensors(test_set).batch(1)
    # # model = models.load_model(checkpoint_filepath)
    # # model.evaluate(dataset.batch(BATCHSIZE))
    # model.load_weights(checkpoint_filepath)
    # test_result = model.predict(test_set)
    # print(test_result)
    # # print(test_result.shape)
    # # 反归一化
    # test_result = (test_result[0]*(label_max - label_min)) + label_min
    # print(test_result)
    # result_path = 'c:\\Users\\caizhi.zheng\\code\\For AI\\loss_picture/result' + suffix + '.csv'
    # with open(result_path, 'w') as f:
    #     np.savetxt(f, test_result, delimiter=',')

