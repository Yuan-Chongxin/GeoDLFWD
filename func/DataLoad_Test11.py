# -*- coding: utf-8 -*-
"""
Load testing data set

Created on Nov 2021

@author: 袁崇鑫

"""

import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io
import pandas as pd
import cv2
import os


def DataLoad_Test(test_size,test_data_dir,data_dim,in_channels,model_dim,data_dsp_blk,label_dsp_blk,start,datafilename,dataname,truthfilename,truthname):
    data1_set1 = np.loadtxt( 'I:/Ar-te.txt')

    # data1_set1 = add_gaussian_noise(data1_set1, 3)
    # np.savetxt('tm23-3zsr.txt', data1_set1)
    data1_set2 = np.loadtxt( 'I:/Ap-te.txt')
    data1_set = np.array([data1_set1, data1_set2])
    data1_set = np.transpose(data1_set, (1, 2, 0))

    for k in range(0, in_channels):
        data11_set = np.float32(data1_set[:, :, k])
        data11_set = np.float32(data11_set)
        data11_set = block_reduce(data11_set, block_size=data_dsp_blk, func=decimate)
        data_dsp_dim = data11_set.shape
        data11_set = data11_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
        if k == 0:
            train1_set = data11_set
        else:
            train1_set = np.append(train1_set, data11_set, axis=0)

    filename_label = test_data_dir + 'label_test/' + str(1) + '.bmdl'
    print(filename_label)
    readFile = np.fromfile(filename_label, dtype=np.float32)
    sagment = readFile[0: len(readFile)]
    sagment = np.reshape(sagment, (256, 1024))
    data2_set = cv2.resize(sagment, (data_dim[1], data_dim[0]), interpolation=cv2.INTER_LINEAR)
    # data2_set = data2_set + random.gauss(0, 0.1) #加噪声
    # Label downsampling
    data2_set = block_reduce(data2_set, block_size=label_dsp_blk, func=np.max)
    label_dsp_dim = data2_set.shape
    data2_set = data2_set.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
    data2_set = np.float32(data2_set)

    test_set = train1_set
    label_set = data2_set



    test_set = test_set.reshape((test_size, in_channels, data_dsp_dim[0] * data_dsp_dim[1]))
    label_set = label_set.reshape((test_size, label_dsp_dim[0] * label_dsp_dim[1]))

    return test_set, label_set, data_dsp_dim, label_dsp_dim

# downsampling function by taking the middle value
def decimate(a,axis):
    idx = np.round((np.array(a.shape)[np.array(axis).reshape(1,-1)]+1.0)/2.0-1).reshape(-1)
    downa = np.array(a)[:,:,idx[0].astype(int)]
    return downa

def updateFile(file,old_str,new_str):
    """
    将替换的字符串写到一个新的文件中，然后将原文件删除，新文件改为原来文件的名字
    :param file: 文件路径
    :param old_str: 需要替换的字符串
    :param new_str: 替换的字符串
    :return: None
    """
    with open(file, "r", encoding="utf-8") as f1,open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if old_str in line:
                line = line.replace(old_str, new_str)
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)