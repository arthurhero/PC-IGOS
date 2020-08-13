'''
convenient tools for conv layers and other
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def l2normalize(w, eps=1e-12):
    # normalize w where w is a pytorch tensor
    return w / (w.norm() + eps)

def init_weights(m):
    # Initialize parameters
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def freeze_params(m):
    for param in m.parameters():
        param.requires_grad = False

def unfreeze_params(m):
    for param in m.parameters():
        param.requires_grad = True

def calc_padding(ker_size,dilate_rate):
    '''
    calculate how much padding is needed for 'SAME' padding
    assume square square kernel
    assume odd kernel size
    '''
    ker_size=(ker_size-1)*(dilate_rate-1)+ker_size
    margin=(ker_size-1)//2
    return margin

def recover_img(tensor):
    '''
    Recover an RGB(A) from a pytorch tensor
    '''
    img=tensor.cpu().detach()
    img=(img+1)*128
    img=img.clamp(0,255)
    img=img.permute(1,2,0)
    img=img.numpy()
    img=img.astype(np.uint8)
    return img
