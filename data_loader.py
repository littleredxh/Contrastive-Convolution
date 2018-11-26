import torch
import torch.utils.data as data
import torchvision
import numpy as np
import os
from scipy.misc import imread, imresize
from PIL import Image
import cv2
'''
Custom DataLoader class for reading pairs from the LFW dataset
'''
class DataLoader(data.Dataset):
    def __init__(self, data_path, trainval, transform):
        self.data_path = data_path
        self.trainval = trainval
        self.transform = transform
        self.data_pair_1, self.data_pair_2, self.label = self.__dataset_info()

    def __getitem__(self, index):
        x = imread(os.path.join(self.data_pair_1[index]), mode="RGB")
        x = Image.fromarray(x) 
        
        y = imread(os.path.join(self.data_pair_2[index]), mode="RGB")
        y = Image.fromarray(y)
        
        label = self.label[index]

        data_pair_1 = self.transform(x)
        data_pair_2 = self.transform(y)
        return data_pair_1, data_pair_2, label

    def __len__(self):
        return len(self.data_pair_1)    

    def __dataset_info(self):
        data_pairs_1 = []
        data_pairs_2 = []
        labels = []
        fname = self.trainval
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip() for x in content] 

        for i, line in enumerate(content):
            if i == 0:
                continue
            line_split = line.split('\t')
            
            if len(line_split) == 3:
                img_name_1 = line_split[0]+'_'+'0'*(4-len(line_split[1]))+line_split[1]+".jpg"
                img_name_2 = line_split[0]+'_'+'0'*(4-len(line_split[2]))+line_split[2]+".jpg"
                img_1 = os.path.join(self.data_path, line_split[0], img_name_1)
                img_2 = os.path.join(self.data_path, line_split[0], img_name_2)
                labels.append([0, 1])
            else:
                img_name_1 = line_split[0]+'_'+'0'*(4-len(line_split[1]))+line_split[1]+".jpg"
                img_name_2 = line_split[2]+'_'+'0'*(4-len(line_split[3]))+line_split[3]+".jpg"
                labels.append([1, 0])
                img_1 = os.path.join(self.data_path, line_split[0], img_name_1)
                img_2 = os.path.join(self.data_path, line_split[2], img_name_2)
            data_pairs_1.append(img_1)
            data_pairs_2.append(img_2)
        return np.array(data_pairs_1), np.array(data_pairs_2), np.array(labels)
    
