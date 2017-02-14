caffe_root = '/vision/u/liyues/software/caffe/'  
import sys  
sys.path.insert(0, caffe_root + 'python')  
import os  
import cv2  
import numpy as np  
import h5py
from scipy import io
import matplotlib.pyplot as plt

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    
def processImage(imgs):
    """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        m = img.mean()
        s = img.std()
        imgs[i] = (img - m) / s
    return imgs

def readdata(filepath):  
    fr=open(filepath,'r')  
    filesplit=[]  
    for line in fr.readlines():  
        s=line.split()  
        s[1:]=[float(x) for x in s[1:]]  
        filesplit.append(s)  
    fr.close()  
    return  filesplit  

def sqrtimg(img):  
    height,width = img.shape[:2]  
    maxlength = max(height,width)  
    sqrtimg0 = np.zeros((maxlength,maxlength,3),dtype='uint8')  
  
    sqrtimg0[round(maxlength*.5-height*.5):round(maxlength*.5+height*.5),round(maxlength*.5-width*.5):round(maxlength*.5+width*.5),:] = img  
    return  sqrtimg0  
  
def generate_hdf5():
    file_path  = '/vision/u/liyues/dataset/rgb/hico/hico_20150920'
    boxsize = 368
    
    anno_file = '%s/anno.mat' %(file_path)
    anno = io.loadmat(anno_file)
    list_test = anno['list_test']
    
    verb_file = '%s/verb_anno_test.npy' %(file_path)
    verb_anno_test = np.load(verb_file)
    

     
    F_imgs = []  
    F_labels = []  
    

    for i in range(list_test.shape[0]):  
        img_name = list_test[i][0][0]
        img_path = '%s/images/test2015/%s' %(file_path, img_name)  

        img = cv2.imread(img_path)
        # img = sqrtimg(img)
        img = cv2.resize(img, (boxsize, boxsize), interpolation = cv2.INTER_CUBIC)
        img = np.float32(img)/256 -0.5
        img = np.transpose(img, (2,0,1))
        
        # img = processImage(img)
        # print(img.shape)
        # print img
        # print np.max(img), np.min(img)
 
        f_label = verb_anno_test[:,i]

        F_imgs.append(img)  
        F_labels.append(f_label)  

  
    F_imgs, F_labels = np.asarray(F_imgs), np.asarray(F_labels)  
    print F_imgs
    print np.sum(F_labels,1)
    # F_imgs = processImage(F_imgs) 
    shuffle_in_unison_scary(F_imgs, F_labels)
  
    with h5py.File(os.getcwd()+ '/test_data.h5', 'w') as f:  
        f['data'] = F_imgs.astype(np.float32)  
        f['labels'] = F_labels.astype(np.float32)  
 
    with open(os.getcwd() + '/test.txt', 'w') as f:  
        f.write(os.getcwd() + '/test_data.h5\n')  
    print i

if __name__ == '__main__':  
    generate_hdf5()
