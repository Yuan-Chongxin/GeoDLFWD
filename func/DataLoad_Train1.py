# -*- coding: utf-8 -*-
"""
读取训练数据

创建于2021年7月

作者：ycx

"""

import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io
import pandas as pd
from IPython.core.debugger import set_trace
from scipy.interpolate import lagrange
import os
import cv2


def DataLoad_Train(train_size, train_data_dir, data_dim, in_channels, model_dim, data_dsp_blk, label_dsp_blk, start,
                   datafilename, dataname, truthfilename, truthname):
    for i in range(start, start + train_size):

        filename_label = train_data_dir + 'input_train/' + str(i) + '.bmdl'

        readFile = np.fromfile(filename_label, dtype=np.float32)
        sagment = readFile[0: len(readFile)]
        sagment = np.reshape(sagment, (256, 1024))
        data2_set = cv2.resize(sagment, (data_dim[1], data_dim[0]), interpolation=cv2.INTER_LINEAR)
        # data2_set = np.reshape(sagment, (data_dim[0],data_dim[1]))
        # Label downsampling
        data2_set = block_reduce(data2_set, block_size=label_dsp_blk, func=np.max)
        data_dsp_dim = data2_set.shape
        data2_set = data2_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
        data2_set = np.float32(data2_set)


        # 读取label
        filename_seis = train_data_dir + 'label_train/' + str(i)
        print(filename_seis)
        # readFile = np.fromfile(filename_seis, dtype=np.float32)
        # data_set1 = readFile[0: len(readFile)]
        # data_set1 = np.reshape(data_set1, (128, 32, 4))
        # data_set1 = np.transpose(data_set1, (2, 1, 0))
        # data1_set = np.array([data_set1[0], data_set1[1]])
        updateFile(filename_seis + '.dat', 'P', '')  # 替换'P'字符
        file = np.loadtxt(filename_seis + '.dat')
        data1_set1 = file[:, 15]
        data1_set1 = np.array(data1_set1).reshape(83, 30).T
        data1_set1 = cv2.resize(data1_set1, (data_dim[1], data_dim[0]), interpolation=cv2.INTER_LINEAR)
        data1_set2 = file[:, 17]
        data1_set2 = np.array(data1_set2).reshape(83, 30).T
        data1_set2 = cv2.resize(data1_set2, (data_dim[1], data_dim[0]), interpolation=cv2.INTER_LINEAR)
        '''
        # 读取视电阻率和相位
        file = np.loadtxt(filename_seis+'.txt')
        data1_set1 = file[:, 11]
        data1_set1 = np.array(data1_set1).reshape(data_dim[1], data_dim[0]).T
        data1_set2 = file[:, 13]
        data1_set2 = np.array(data1_set2).reshape(data_dim[1], data_dim[0]).T
        '''
        '''
        mmd = open(filename_seis + '.csv')
        data1_set = pd.read_csv(mmd,header=None)
        # 读取视电阻率和相位
        data1_set1 = data1_set.loc[:, [11]]
        data1_set1 = np.array(data1_set1).reshape(data_dim[1], data_dim[0]).T

        data1_set2 = data1_set.loc[:, [13]]
        data1_set2 = np.array(data1_set2).reshape(data_dim[1], data_dim[0]).T

        data1_set3 = data1_set.loc[:, [15]]
        data1_set3 = np.array(data1_set3).reshape(data_dim[1], data_dim[0]).T

        data1_set4 = data1_set.loc[:, [17]]
        data1_set4 = np.array(data1_set4).reshape(data_dim[1], data_dim[0]).T

        data1_set = np.array([data1_set1,data1_set2,data1_set3,data1_set4])

        data1_set = np.array([data1_set1, data1_set2])
        #data1_set = np.float32(data1_set.reshape([data_dim[0], data_dim[1], in_channels]))
        '''
        # data1_set = np.array([data1_set1, data1_set2])
        data1_set = np.array([data1_set1, data1_set2])
        '''
        data1_set1 = block_reduce(data1_set1, block_size=label_dsp_blk, func=np.max)
        data_dsp_dim = data1_set1.shape
        train1_set = data1_set1.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
        train1_set = np.float32(train1_set)
        '''
        # 改变维度 [h, w, c] --> [c, h, w]
        # data1_set = np.transpose(data1_set, (1,2,0))
        data1_set = np.transpose(data1_set, (1, 2, 0))

        for k in range(0, 2):
            data11_set = np.float32(data1_set[:, :, k])
            data11_set = np.float32(data11_set)
            # Data downsampling
            # note that the len(data11_set.shape)=len(block_size.shape)=2
            data11_set = block_reduce(data11_set, block_size=label_dsp_blk, func=decimate)
            label_dsp_dim = data11_set.shape
            data11_set = data11_set.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
            if k == 0:
                train1_set = data11_set
            else:
                train1_set = np.append(train1_set, data11_set, axis=0)

        if i == start:
            train_set = data2_set
            label_set = train1_set
        else:
            train_set = np.append(train_set, data2_set, axis=0)
            label_set = np.append(label_set, train1_set, axis=0)
    # train_set = get_normal_data(train_set)
    # label_set = get_normal_data(label_set)
    train_set = train_set.reshape((train_size, in_channels, data_dsp_dim[0] * data_dsp_dim[1]))
    label_set = label_set.reshape((train_size, 2, label_dsp_dim[0] * label_dsp_dim[1]))

    return train_set, label_set, data_dsp_dim, label_dsp_dim


# downsampling function by taking the middle value
def decimate(a, axis):
    idx = np.round((np.array(a.shape)[np.array(axis).reshape(1, -1)] + 1.0) / 2.0 - 1).reshape(-1)
    downa = np.array(a)[:, :, idx[0].astype(int), idx[1].astype(int)]
    return downa


def updateFile(file, old_str, new_str):
    """
    将替换的字符串写到一个新的文件中，然后将原文件删除，新文件改为原来文件的名字
    :param file: 文件路径
    :param old_str: 需要替换的字符串
    :param new_str: 替换的字符串
    :return: None
    """
    with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if old_str in line:
                line = line.replace(old_str, new_str)
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)


def get_normal_data(data1):
    amin = 0.1
    amax = 1000000
    return (data1 - amin) / (amax - amin)
