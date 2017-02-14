# import cv2 as cv 
import numpy as np
import scipy
# import PIL.Image
import math
# import caffe
# import time
# from config_reader import config_reader
import util
import copy
# import matplotlib
# %matplotlib inline
# import pylab as plt
from scipy import io

anno_file = '/vision/u/liyues/dataset/rgb/hico/hico_20150920/anno.mat'
anno = io.loadmat(anno_file)

list_action = anno['list_action']
hoi_list = []
for i in range(list_action.shape[0]):
    hoi_list.append(list_action[i]['vname'][0][0])
hoi_list = np.array(hoi_list)

verb_list = []
with open('hico_list_vb.txt', 'r') as f:
    counter = 0
    for line in f.readlines():
        counter = counter +1
        if(counter>2):
            verb_list.append(line.split(' ')[2])
verb_list = np.array(verb_list)

## train annotation
anno_train = anno['anno_train']
bin_train = np.zeros(anno_train.shape)
bin_train[anno_train == 1] = 1
list_action
verb_anno_train = np.zeros([verb_list.shape[0],anno_train.shape[1]])
for j in range(verb_list.shape[0]):
    hoi_lines = bin_train[ hoi_list==verb_list[j], :]
    verb_anno_train[j,:] = np.sum(hoi_lines,0)
np.save('verb_anno_train',verb_anno_train)

## test annotation
anno_test = anno['anno_test']
bin_test = np.zeros(anno_test.shape)
bin_test[anno_test == 1] = 1

verb_anno_test = np.zeros([verb_list.shape[0],anno_test.shape[1]])
for j in range(verb_list.shape[0]):
    hoi_lines = bin_test[ hoi_list==verb_list[j], :]
    verb_anno_test[j,:] = np.sum(hoi_lines,0)
np.save('verb_anno_test',verb_anno_test)