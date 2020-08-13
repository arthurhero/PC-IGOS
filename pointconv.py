import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

from utils.tools import *
from modelnetdataset import *
from pointconv_utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class DensityNet(nn.Module):
    '''
    Calculate density scale of points
    input - density, B x 1 x K x N
    output - transformed density, B x 1 x K x N
    '''
    def __init__(self):
        super(DensityNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.sigmoid(self.conv2(x))
        return x

class WeightNet(nn.Module):
    '''
    Get the weights according to the point positions
    input - transalted nearest neighbor position, B x 3 x K x N
    output - weights, B x Cout x K x N
    '''
    def __init__(self, in_channels, out_channels):
        super(WeightNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x


class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, sigma, fix_fps):
        super(PointConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.linear = nn.Linear(in_channels * 32, out_channels)
        self.weightnet = WeightNet(3, 32)
        self.densitynet = DensityNet()
        self.sigma=sigma
        self.fix_fps=fix_fps
        
        self.weightnet.apply(init_weights)
        self.densitynet.apply(init_weights)

    def forward(self, inputs, inputs_pos, inverse_density):
        '''
        input - all points in dataset, B x C x N
        input_pos - B x 3 x N
        inverse_density - B x 1 x N
        '''
        b, c, n = inputs.size()
        out_num = n//self.stride
        # downsample the points
        if n>out_num:
            sample_idx = fps_sampling(input_pos, out_num, self.fix_fps) # get indices of representative points - B x n
            sample_idx_f = sample_idx.unsqueeze(dim=1).expand(-1, c, -1) # B x c x n
            sample_idx_p = sample_idx.unsqueeze(dim=1).expand(-1, 3, -1) # B x 3 x n
            sampled_inputs = inputs.gather(dim=-1, index=sample_idx_f).contiguous().view(b, c, -1) # B x c x n
            sampled_pos = inputs_pos.gather(dim=-1, index=sample_idx_p).contiguous().view(b, 3, -1) # B x 3 x n
            inputs = sampled_inputs
            inputs_pos = sampled_pos

        n=out_num

        pairwise_dist = pairwise_distance(inputs_pos)
        pairwise_dist = pairwise_dist.contiguous()
        nn_idx = knn(pairwise_dist, k = self.kernel_size) # nearest neighbor index - B x K x N
        nn_pos = gather_nd(inputs_pos, nn_idx) # nearest neighbor position - B x 3 x K x N
        nn_inputs = gather_nd(inputs, nn_idx) # nearest neighbor feature - B x C x K x N

        lnn_pos = nn_pos - inputs_pos.unsqueeze(dim=2) # transalted nearest neighbor position - B x 3 x K x N
        nn_inverse_density = gather_nd(inverse_density, nn_idx) # density of neighbors - B x 1 x K x N
        inverse_max_density,_ = nn_inverse_density.max(dim=2,keepdim=True) # B x 1 x 1 x N
        density_scale = nn_inverse_density.div(inverse_max_density) # B x 1 x K x N

        density_scale = self.densitynet(density_scale)
        #apply inverse density scale to features
        scaled_nn_inputs = nn_inputs.mul(density_scale)

        weights = self.weightnet(lnn_pos.float()) # B x Bout x K x N

        new_feat = torch.matmul(scaled_nn_inputs.permute(0, 3, 1, 2), weights.permute(0, 3, 2, 1)).view(b, n, -1) # B x N x (Cin x Bout)
        new_feat = self.linear(new_feat).permute(0, 2, 1) # B x Cout x N

        return new_feat

class PointMaxPool(nn.Module):
    def __init__(self, target_num):
        super(PointMaxPool, self).__init__()
        self.target_num = target_num

    def forward(self, inputs,inputs_pos):
        b, c, n = inputs.size()
        out_num = self.target_num
        stride=n//out_num
        kernel_size=stride
        if n>out_num:
            sample_idx = fps_sampling(inputs_pos, out_num) # get indices of representative points - B x n
            sample_idx_p = sample_idx.unsqueeze(dim=1).expand(-1, 3, -1) # B x 3 x n
            sampled_pos = inputs_pos.gather(dim=-1, index=sample_idx_p).contiguous().view(b, 3, -1) # B x 3 x n
        else:
            sampled_pos = inputs_pos

        pairwise_dist = pairwise_distance_general(sampled_pos, inputs_pos)
        nn_idx = knn(pairwise_dist, k = kernel_size)
        nn_inputs = gather_nd(inputs, nn_idx) # nearest neighbor feature - B x C x K x n
        nn_inputs, _ = torch.max(nn_inputs, dim = 2) # max nn feature - B x C x n

        inputs = nn_inputs 
        inputs_pos = sampled_pos
        return inputs,inputs_pos

class PointAvgPool(nn.Module):
    def __init__(self,target_num,fix_fps=False):
        super(PointAvgPool, self).__init__()
        self.target_num = target_num 
        self.fix_fps=fix_fps

    def forward(self, inputs, inputs_pos):
        b, c, n = inputs.size()
        out_num = self.target_num
        stride=n//out_num
        kernel_size=stride
        # downsample the points
        if n>out_num:
            sample_idx = fps_sampling(inputs_pos, out_num, self.fix_fps) # get indices of representative points - B x n
            sample_idx_p = sample_idx.unsqueeze(dim=1).expand(-1, 3, -1) # B x 3 x n
            sampled_pos = inputs_pos.gather(dim=-1, index=sample_idx_p).contiguous().view(b, 3, -1) # B x 3 x n
        else:
            sampled_pos = inputs_pos

        pairwise_dist = pairwise_distance_general(sampled_pos, inputs_pos)
        nn_idx = knn(pairwise_dist, k = kernel_size)
        nn_inputs = gather_nd(inputs, nn_idx) # nearest neighbor feature - B x C x K x n
        nn_inputs = torch.mean(nn_inputs, dim = 2) # max nn feature - B x C x n

        inputs = nn_inputs 
        inputs_pos = sampled_pos
        return inputs,inputs_pos

class SpiderPC(nn.Module):
    def __init__(self,in_channels,num_class,sample_num,fix_fps=False):
        '''
        in_channels - number of input channels
        num_class - number of classes to be classified
        sample_num - number of points
        '''
        super(SpiderPC, self).__init__()
        self.conv1=PointConv(in_channels, 32, 20, 1, 0.05,fix_fps)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2=PointConv(32, 64, 20, 1, 0.05,fix_fps)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3=PointConv(64, 128, 20, 1, 0.05,fix_fps)
        self.bn3 = nn.BatchNorm1d(128)
        self.avgpool=PointAvgPool(2,fix_fps)
        self.linear1= nn.Linear(224*2,512)
        self.linear2= nn.Linear(512,256)
        self.linear3= nn.Linear(256,num_class)

    def forward(self, x_pos, x=None):
        if x is None:
            x=x_pos.clone()
        b=x.shape[0]
        inverse_density=get_inverse_density(x_pos,k=20,sigma=0.5)
        x1 = F.relu(self.bn1(self.conv1(x, x_pos,inverse_density))) # B x 32 x 1024
        x2 = F.relu(self.bn2(self.conv2(x1, x_pos,inverse_density))) # B x 64 x 1024
        x3 = F.relu(self.bn3(self.conv3(x2, x_pos,inverse_density))) # B x 128 x 1024
        x = torch.cat([x1,x2,x3],dim=1)
        x, x_pos = self.avgpool(x, x_pos) # B x 224 x 2
        x = x.clone().view(b,-1)
        x = self.linear1(x)
        x = self.linear2(x)
        out = self.linear3(x)
        return out
