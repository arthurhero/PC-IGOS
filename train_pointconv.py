import os
import sys 

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

from modelnetdataset import * 
from pointconv import * 

#hyperparams
batch_size=32
num_class=40
sample_num=1024
epoch=100
lr=0.0001
theta=0.95 #ratio of points to keep

root_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_folder, 'utils'))
log_folder=os.path.join(root_folder,'log')
ckpt_path=os.path.join(log_folder,'model40_ap.ckpt')

#if want to use random dropout, use this 
def my_collate(batch):
    '''
    batch - [item]
    item - (pc,target) or (pc,target,blurs)
    pc - 6 x n
    target - long scalar
    blurs - 10 x 3 x n
    '''
    n=batch[0][0].shape[1]
    perm = torch.randperm(n)
    drop_ratio = perm.new(1).float()
    drop_ratio.uniform_(to=theta)
    keep_num=round((n-(n*drop_ratio)).item())
    keep_idx = perm[:keep_num]
    data = [item[0][:,keep_idx] for item in batch]
    target = [item[1] for item in batch]
    target = torch.stack(target)
    data = torch.stack(data)
    return (data, target)

dataset=Model40DataSet(model10=False, train=True, normal=False, augmentation=True, sample_num=sample_num)
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=8)

test_dataset=Model40DataSet(model10=False, train=False, normal=False, augmentation=False)
test_dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=batch_size)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train():
    pcnet=SpiderPC(3,num_class,sample_num).to(device)
    criterion = nn.CrossEntropyLoss()

    pcnet.train()
    optimizer = torch.optim.Adam(pcnet.parameters(),lr=lr,betas=(0.5,0.9))

    #load checkpoint
    if os.path.isfile(ckpt_path):
        pcnet.load_state_dict(torch.load(ckpt_path))
        print("Loaded ckpt!")

    for e in range(epoch):
        step=0
        correct=0.0
        total=0.0
        for i,(pc_batch,targets) in enumerate(dataloader):
            pc_batch, targets = pc_batch.to(device), targets.to(device)
            optimizer.zero_grad()
            input_pos = pc_batch[:,:3,:]
            inputs = pc_batch[:,3:,:]
            outputs = pcnet(inputs,input_pos)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            correct+=predicted.eq(targets).sum().item()
            total+=targets.size(0)
            if step%100==99:
                torch.save(pcnet.state_dict(), ckpt_path)
                acc=correct/total
                correct=0.0
                total=0.0
                print('Epoch [{}/{}] , Step {}, Loss: {:.4f}, Accuracy: {:.4f}'
                            .format(e+1, epoch, step, loss.item(), acc))
            step+=1
        if (e+1)%5==0:
            test(pcnet)

def test(pcnet=None):
    if pcnet==None:
        pcnet=SpiderPC(3,num_class,sample_num)
        pcnet=pcnet.to(device)
        #load checkpoint
        if os.path.isfile(ckpt_path):
            pcnet.load_state_dict(torch.load(ckpt_path))
            print("Loaded ckpt!")
    pcnet.eval()
    criterion = nn.CrossEntropyLoss()
    correct=0.0
    total=0.0
    for i,(pc_batch,targets) in enumerate(test_dataloader):
        pc_batch, targets = pc_batch.to(device), targets.to(device)
        input_pos=pc_batch[:,:3,:]
        inputs = pc_batch[:,3:,:]
        outputs = pcnet(inputs,input_pos)
        loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct+=predicted.eq(targets).sum().item()
        total+=targets.size(0)
    acc=correct/total
    print('TESTING Loss: {:.4f}, Accuracy: {:.4f}'
            .format(loss.item(),acc))
    pcnet.train()


train()
#test()
