import functools
import logging
import bisect
import random
import pickle

import torch.utils.data as data
import cv2
import numpy as np
import glob
from concern.config import Configurable, State
import math
import scipy.io
#import data.data_utils
#from data.data_utils import generate_rbox

def preprocess_words(word_ar):
    words = []
    #print("word_ar = ", word_ar)
    for ii in range(np.shape(word_ar)[0]):
        s = word_ar[ii]
        start = 0
        while s[start] == ' ' or s[start] == '\n':
            start += 1
        for i in range(start + 1, len(s) + 1):
            if i == len(s) or s[i] == '\n' or s[i] == ' ':
                if start != i:
                    words.append(s[start : i])
                start = i + 1
    return words

class PufDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    mat_list = State()
    processes = State(default=[])
    mode = State(default="train")
    snumber= State(default=None)
    seqList= State(default=False)
    data_ratio= State(default="R811")
    nList= State(default=True)

    def __init__(self, data_dir=None, mat_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.mat_list = mat_list or self.mat_list
        
        if 'train' in self.mat_list[0]:
            self.is_training = False
        else:
            self.is_training = False
            
        self.debug = cmd.get('debug', False)
        self.debug = False
        
        self.image_paths = []
        self.gt_maps = []
        self.gt_maps_char = []
        
        self.get_all_samples(self.snumber)
        
        #self.nList=True
        if (self.seqList):
            self.map_idx=range(0, self.num_samples)
        elif (self.nList):
            self.map_idx= random.sample(range(0, len(self.image_paths)), self.num_samples)
            with open("synthrndlist.txt", 'wb') as f:
                pickle.dump((self.map_idx), f)
            f.close()
        else:
            with open("synthrndlist.txt", 'rb') as f:
                self.map_idx = pickle.load(f)
            f.close()
            if len(self.map_idx) != self.num_samples:
                print('Synthdata exist list size is not expected number (', self.num_samples, ')')
                quit()
        
        #self.seqList=True
        if (self.seqList):
            self.map_idx = list(range(0,self.num_samples))
        #data_length = self.num_samples


        if self.data_ratio=="R811":
            self.train_s=0
            self.train_e=int(self.num_samples*8/10)
        
            self.valid_s=self.train_e
            self.valid_e=self.train_e+int(self.num_samples/10)
        
            self.test_s =self.valid_e
            self.test_e =self.num_samples
        else:
            self.train_s=0
            self.train_e=int(self.num_samples)
        
            self.valid_s=0
            self.valid_e=int(self.num_samples)
        
            self.test_s =0
            self.test_e =int(self.num_samples)


    def get_all_samples(self, snumber):
        for i in range(len(self.data_dir)):
            print("Load Mat file ", self.data_dir)
            mat = scipy.io.loadmat(self.mat_list[i])
            print("Done...")
            image_list = mat['imnames'][0]            
            image_path = [self.data_dir[i]+timg[0] for timg in image_list]
            self.image_paths += image_path


        if(snumber== None or snumber > len(self.image_paths)):
            self.num_samples = len(self.image_paths)
        else:
            self.num_samples = snumber
        
        
        
#        self.targets = self.load_ann()
#        self.targets_char = self.load_ann_char()
        
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)



        
    def __getitem__(self, index, retry=0):
        
        #index=5731
        if index >= self.__len__():
            index = index % self.__len__()
            
        if (self.mode == 'train'):
            index_update = index
        elif (self.mode == 'valid'):
            index_update = self.valid_s + index
        else:
            index_update = self.test_s + index
         
        
        index_update = self.map_idx[index_update]
        #index_update = 1181
            
        data = {}
        image_path = self.image_paths[index_update]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')
        data['index']=index_update

        if self.debug== True:
            print("index = ",index_update," Image file name: ", image_path,"\n")
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img[:432]
        
        #print ("image size=", img.shape)
        #print("Start index process ", index)
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
            
            #score_map, geo_map, training_mask, rects=generate_rbox((data['image'].shape[1], data['image'].shape[0]), data['polygons'], data['lines_text'])
            #data['score_map']=score_map
            #data['geo_map']=geo_map
            #data['training_mask']=training_mask
            #data['rects']=rects
        #print("End index process, image size =", data['image'].shape, "  index = ", index)
        #print("End index process, len(data) =", len(data))
        
        #score_map, geo_map, training_mask = generate_rbox((img.shape[1], img.shape[0]), data['lines']['poly'], data['lines']['text'])
        
        #print (data.keys())
        return data['image'], 0

    def __len__(self):
        
        if (self.mode == 'train'):
            return self.train_e - self.train_s
        elif (self.mode == 'valid'):
            return self.valid_e - self.valid_s
        else:
            return self.test_e - self.test_s
        
        #return len(self.image_paths)
        #return len(self.image_paths)
