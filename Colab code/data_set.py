import six.moves.cPickle as Pickle
import torch as th
import cv2
import numpy as np
from random import choice

def loadImage(path):
    inImage_ = cv2.imread(path)
    #inImage = inImage.astype('uint8')
    inImage = cv2.cvtColor(inImage_, cv2.COLOR_RGB2BGR)
    info = np.iinfo(inImage.dtype)
    inImage = inImage.astype(np.float) / info.max

    iw = inImage.shape[1]
    ih = inImage.shape[0]
    if iw < ih:
        inImage = cv2.resize(inImage, (64, int(64 * ih/iw)))
    else:
        inImage = cv2.resize(inImage, (int(64 * iw / ih), 64))
    inImage = inImage[0:64, 0:64]
    return th.from_numpy(2 * inImage - 1).transpose(0, 2).transpose(
        1, 2
    )


class LookbookDataset():
    def __init__(self, data_dir, index_dir):
        self.data_dir = data_dir
        with open(index_dir+'new_cloth_table.pkl', 'rb') as cloth:
            self.cloth_table = Pickle.load(cloth)
        with open(index_dir+'new_model_table.pkl', 'rb') as model:
            self.model_table = Pickle.load(model)
        with open(index_dir+'new_user_table.pkl', 'rb') as user:
            self.user_table = Pickle.load(user)

        #self.cloth_table = (sorted(self.cloth_table))
        #self.model_table = (sorted(self.model_table))
        #self.user_table1 = [x for x in self.user_table if x]
        #self.user_table1 = (sorted(self.user_table1))

        self.cn = len(self.cloth_table)
        self.path = data_dir

    def getbatch(self, batchsize):
        batch1 = []
        batch2 = []
        batch3 = []
        batch4 = []
        for i in range(batchsize):
            seed = th.randint(1, 100000, (1,)).item()
            th.manual_seed((i+1)*seed)
            #r1 = th.randint(0, self.cn, (1,)).item()
            #r2 = th.randint(0, self.cn, (1,)).item()
            r1 = th.randint(0, 255, (1,)).item()
            #r1 = choice([i for i in range(0,331) if i not in range(101,176)])
            r1 = int(r1)
            
            r2 = th.randint(0, 9, (1,)).item()
            r2 = int(r2)
            mn = len(self.model_table[r1])
            #r3 = th.randint(0, mn, (1,)).item()
            r3 = th.randint(0, 1, (1,)).item()
            r3 = int(r3)
			
            #print(r1,r2,r3)
            
            path1 = self.cloth_table[r1]
            #path2 = self.cloth_table[r2]
            try:
              path2 = self.model_table[r1][r2]
            except:
              path2 = self.model_table[r1][0]
              
            try:  
              path3 = self.model_table[r1][r3]
            except:
              path3 = self.model_table[r1][0]
            
            try:
              path4 = self.user_table[r1][r3]
            except:
              path4 = self.user_table[r1][0]
            
            #print(self.path+path1)
            img1 = loadImage(self.path + path1)
            img2 = loadImage(self.path + path2)
            img3 = loadImage(self.path + path3)
            img4 = loadImage(self.path + path4)
            batch1.append(img1)
            batch2.append(img2)
            batch3.append(img3)
            batch4.append(img4)
        return th.stack(batch2), th.stack(batch3), th.stack(batch4), th.stack(batch1)