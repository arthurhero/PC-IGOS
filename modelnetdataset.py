import os
import os.path
import numpy as np
import sys

import torch
from torch.utils.data import Dataset, DataLoader

from pointconv_utils import *
from blur_utils import plane_blur 

root_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_folder, 'utils'))

dataset_folder=os.path.join(root_folder,'data/modelnet40_normal_resampled')

def load_flist_file(f,shape=None):
    flist = [line.rstrip('\n') for line in open(f)]
    return flist

class Model40DataSet(Dataset):
    '''
    Model40 point cloud dataset sampled by Charles R. Qi et. al.
    model10 - use Model10 or not
    train - use train or test
    normal - use normal as feature or pos as feature
    shape - whether only load a specific shape (0-39)
    entry - whether only load a specific entry  (e.g. 'bed_0595')
    save_blur - precompute the 10 smoothed shapes while reading data and save to disk
    load_blur - whether load the saved smoothed shapes along with the original shape
                (if no saved shape available, will compute on real-time and save)
    '''
    def __init__(self, model10=False, train=True, normal=False,augmentation=False,sample_num=1024,shape=None,entry=None,save_blur=False,load_blur=False):
        root = dataset_folder
        shapelist_f=''
        flist_f = ''
        if model10:
            shapelist_f = os.path.join(root,'modelnet10_shape_names.txt')
            if train:
                flist_f = os.path.join(root,'modelnet10_train.txt')
            else:
                flist_f = os.path.join(root,'modelnet10_test.txt')
        else:
            shapelist_f = os.path.join(root,'modelnet40_shape_names.txt')
            if train:
                flist_f = os.path.join(root,'modelnet40_train.txt')
            else:
                flist_f = os.path.join(root,'modelnet40_test.txt')
        shape_names=load_flist_file(shapelist_f)
        self.shape_dict=dict()
        self.shape_list=shape_names
        i=0
        for s in shape_names:
            self.shape_dict[s]=i
            i=i+1
        self.flist=load_flist_file(flist_f)
        if shape is not None:
            self.flist=list(filter(lambda x : x.rsplit('_',1)[0]==shape_names[shape],self.flist))
        if entry is not None:
            self.entry=entry
        else:
            self.entry=None
        self.normal=normal
        self.augmentation=augmentation
        self.sample_num=sample_num
        self.save_blur=save_blur
        self.load_blur=load_blur

    def __len__(self):
        if self.entry is not None:
            return 1
        else:
            return len(self.flist)

    def __getitem__(self,idx):
        if self.entry is not None:
            entry=self.entry
        else:
            entry=self.flist[idx]
        #print(entry)
        shape=entry.rsplit('_',1)[0]
        shape_index=self.shape_dict[shape]
        target=torch.tensor(shape_index,dtype=torch.long)
        pc_path=os.path.join(dataset_folder,shape+'/'+entry+'.txt')
        if self.save_blur or self.load_blur:
            blur_path=os.path.join(dataset_folder,shape+'/'+entry+'.blur')

        pc_orig = load_pc(pc_path)
        pc_orig=torch.from_numpy(pc_orig) # n x 6
        pc_orig=pc_orig.permute(1,0) # 6 x n 
        pc_orig_pos=pc_orig[0:3]
        pc_orig_pos=preprocess_pc(pc_orig_pos,self.augmentation)
        pc_orig[0:3]=pc_orig_pos
        blurs=None
        while blurs is None:
            #sample n points from point cloud
            perm = torch.randperm(pc_orig.shape[1])
            idx = perm[:self.sample_num]
            pc = pc_orig[:,idx]
            if self.normal == False:
                pc[3:]=pc[0:3]
            if self.save_blur:
                print("blurring",entry)
                blurs=plane_blur(pc[:3].unsqueeze(0),mask=None,single=False)
                if blurs is None:
                    continue
                pc_blurs=torch.cat([pc[:3].unsqueeze(0).unsqueeze(0),blurs],dim=1)
                #print(pc_blurs.shape)
                torch.save(pc_blurs, blur_path)
            if self.load_blur:
                if os.path.isfile(blur_path):
                    pc_blurs=torch.load(blur_path)
                    pc=pc_blurs[0,0] # 3 x n
                    pc=torch.cat([pc,pc],dim=0)
                    blurs=pc_blurs[:,1:] # 1 x 10 x 3 x n
                else:
                    print("blurring",entry)
                    blurs=plane_blur(pc[:3].unsqueeze(0),mask=None,single=False)
                    if blurs is None:
                        continue
                    pc_blurs=torch.cat([pc[:3].unsqueeze(0).unsqueeze(0),blurs],dim=1)
                    torch.save(pc_blurs, blur_path)
                return (pc,target,blurs[0])
            else:
                return (pc,target)

def save_all_blurs(shape=None,entry=None):
    dataset=Model40DataSet(model10=False, train=False, normal=False,augmentation=False,sample_num=1024,shape=shape,entry=entry,save_blur=True)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=1)
    for i,(pc_batch,targets) in enumerate(dataloader):
        continue

if __name__ == '__main__':
    save_all_blurs()
    #save_all_blurs(shape=39)
    #save_all_blurs(entry='bed_0595')
