import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F 

from utils.tools import *
from open3d import utility,visualization,geometry

def load_pc(f):
    points=[line.rstrip() for line in open(f)]
    points=[p.split(',') for p in points]
    points=np.asarray(points, dtype=np.float32) # n x 6
    return points

def normalize_pc(pc_pos):
    '''
    pc_pos - 3 x n
    '''
    mean = pc_pos.mean(1,keepdim=True) # 3
    pc_pos-=mean
    max_dis=pc_pos.pow(2).sum(0).pow(0.5).max()
    #maxi = pc_pos.abs().max()
    pc_pos = pc_pos.div(max_dis)
    return pc_pos

def rotate_pc(pc_pos):
    '''
    pc_pos - 3 x n
    '''
    twopi=2 * 3.1415926
    a = torch.rand(1)*twopi # rotation around x
    b = torch.rand(1)*twopi # rotation around y
    c = torch.rand(1)*twopi # rotation around z
    #only rotate around y
    a[0]=0.0
    c[0]=0.0
    asin=a.sin()
    acos=a.cos()
    bsin=b.sin()
    bcos=b.cos()
    csin=c.sin()
    ccos=c.cos()
    rot_mat = torch.stack([
        torch.stack([bcos*ccos,acos*csin+asin*bsin*ccos,asin*csin-acos*bsin*ccos]),
        torch.stack([-bcos*csin,acos*ccos-asin*bsin*csin,asin*ccos+acos*bsin*csin]),
        torch.stack([bsin,-asin*bcos,acos*bcos])])
    rot_mat=rot_mat.view(3,3)
    pc_pos=rot_mat.matmul(pc_pos)
    return pc_pos

def jitter_pc(pc_pos):
    '''
    pc_pos - 3 x n
    '''
    noise=torch.randn_like(pc_pos)/50.0
    pc_pos+=noise
    return pc_pos

def preprocess_pc(pc_pos,aug=True):
    pc_pos=normalize_pc(pc_pos)
    if aug:
        pc_pos=rotate_pc(pc_pos)
        pc_pos=jitter_pc(pc_pos)
        return pc_pos
    return pc_pos

# visualize a 3 x n point cloud pytorch tensor
def visualize_pc(pc,mask=None):
    '''
    pc - 3 x n
    mask - 1 x n, [0,1]
    '''
    pc = pc.permute(1,0) # n x 3
    pc = pc.cpu().detach().numpy()
    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(pc)
    if mask is not None:
        mask=mask/mask.max()
        #mask=mask.pow(0.5) #dilate the values
        mask = mask.expand(3,-1).permute(1,0) #nx3
        red = mask.new(mask.shape).zero_()
        red[:,0]+=1.0
        green = mask.new(mask.shape).zero_()
        green[:,1]+=1.0
        blue = mask.new(mask.shape).zero_()
        blue[:,2]+=1.0
        less = ((2*mask).mul(green)+(1.0-(2*mask)).mul(blue)).mul(mask.lt(0.5).float())
        more = (((mask-0.5)*2).mul(red)+(1.0-((mask-0.5)*2)).mul(green)).mul(mask.ge(0.5).float())
        mask = less+more
        mask_max,_ = mask.max(dim=1,keepdim=True)
        mask = mask.div(mask_max)
        mask = mask.cpu().detach().numpy()
        pcd.colors = utility.Vector3dVector(mask)
        visualization.draw_geometries([pcd])

def fps_sampling(pc_pos, n, fix_first=False):
    """ Found here: 
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Farthest point sampling
    pc_pos - B x 3 x N, all points in dataset
    n - number of points to be sampled
    fix_first - fix the index of the first sampled point

    return indices of sampled points - B x n
    """
    calc_distances = lambda p0, pts: ((p0.unsqueeze(1).expand(-1,pts.shape[1]) 
        - pts)**2).sum(dim = 0) # p0 - 3, pts - 3 x N

    pc_pos_batch = pc_pos.split(1)

    fps_idx_list=list()

    for pts in pc_pos_batch:
        pts=pts[0] # 3 x N
        N=pts.shape[1]
        farthest_idx = pts.new(n).zero_().long() # n
        if fix_first:
            farthest_idx[0] = 0
        else:
            farthest_idx[0] = torch.randint(N,(1,))
        distances = calc_distances(pts[:,farthest_idx[0]], pts) # N

        for i in range(1, n):
            _, idx = torch.max(distances,0)
            farthest_idx[i] = idx 
            farthest_pts = pts[:,farthest_idx[i]]
            distances = torch.min(distances, calc_distances(farthest_pts, pts))

        farthest_idx=farthest_idx.unsqueeze(0)
        fps_idx_list.append(farthest_idx)

    fps_idx=torch.cat(fps_idx_list)
    return fps_idx

def inverse_density_sampling(inputs, num_samples, k = 20):
    """
    Inverse density sampling
    inputs - B x 3 x N
    num_samples - number of sampled points
    k - number of neighbors when estimating density

    return indices of sampled points - B x n
    """
    n = inputs.size(2)
    #(b, n, n)
    pair_dist = pairwise_distance(inputs)
    if k > n:
        k = n
    pair_dist=pair_dist.contiguous()
    distances, _ = pair_dist.topk(dim=1,k=k,largest=False) # B x K x N

    #(b, n)
    distances_avg = torch.abs(torch.mean(distances, dim=1)) + 1e-8 # B x N
    prob_matrix = distances_avg / torch.sum(distances_avg, dim = 1, keepdim=True) # B x N

    #(b, num_samples)
    sample_idx = torch.multinomial(prob_matrix, num_samples) # B x n
    return sample_idx

def kernel_density_estimation(nn_pos,sigma,normalize = False):
    '''
    Calculate the kernel density estimation using Gaussian kernal and k nearest neighbors of the N points
    nn_pos - B x 3 x K x N, the xyz position of the K neighbors of all the N points relative to the center
    sigma - the bandwidth of the Gaussian kernel
    normalize - normalize it using the largest density among the N points

    return density - B x 1 x N
    '''
    sigma_=nn_pos.new(1).float()
    sigma_[0]=sigma
    sigma=sigma_.clone()
    posdivsig = nn_pos.div(sigma) # x/sig, y/sig, z/sig
    quadform = posdivsig.pow(2).sum(dim=1) # (x^2+y^2+z^2)/sig^2
    #print(quadform) # should be B x K x N
    logsqrtdetSigma = sigma.log() * 3 # log(sigma^3)
    twopi=sigma.clone()
    twopi[0]=2 * 3.1415926
    mvnpdf = torch.exp(-0.5 * quadform - logsqrtdetSigma - 1.5 * torch.log(twopi)) #(2pi)^(-3/2)*sigma^(-3)*exp(-0.5*(x^2+y^2+z^2)/sig^2)
    mvnpdf = torch.sum(mvnpdf, dim = 1, keepdim = True) # sum all neighbors
    #print(mvnpdf) # should be B x 1 x N

    scale = 1.0 / nn_pos.shape[2] #1/K
    density = mvnpdf*scale # B x 1 x N

    if normalize:
        density_max,_ = density.max(dim=2, keepdim=True) # B x 1 x 1
        density = density.div(density_max)

    return density

def pairwise_distance(input_pos):
    """
    Args:
    input_pos: tensor(batch_size, num_dims, num_points)
    Returns:
    pairwise distance: (batch_size, num_points, num_points)
    """
    b = input_pos.size(0)
    input_pos_transpose = input_pos.contiguous().permute(0, 2, 1)
    input_pos_inner = torch.matmul(input_pos_transpose, input_pos)
    input_pos_inner = -2 * input_pos_inner
    input_pos_square = torch.sum(input_pos * input_pos, dim = 1, keepdim=True)
    input_pos_square_transpose = input_pos_square.contiguous().permute(0, 2, 1)
    return input_pos_inner + input_pos_square + input_pos_square_transpose

def pairwise_distance_general(queries, input_pos):
    '''
    Args:
    queries: (batch_size, num_dims, num_points')
    input_pos: tensor(batch_size, num_dims, num_points)
    Returns:
    pairwise distance: (batch_size, num_points, num_points')
    '''
    #(b, n, c)
    input_pos_transpose = input_pos.contiguous().permute(0, 2, 1)
    #(b, n, n')
    inner = torch.matmul(input_pos_transpose, queries)
    inner = -2 * inner
    #(b, n, 1)
    input_pos_square_transpose = torch.sum(input_pos_transpose * input_pos_transpose, dim = 2, keepdim=True)
    #(b, 1, n')
    queries_square =  torch.sum(queries * queries, dim = 1, keepdim=True)
    return queries_square + inner + input_pos_square_transpose

def knn(dist, k=20, ret_dist=False):
    """
    Get KNN based on dist matrix
    Args:
    dist: (batch_size, num_points, num_points)
    k:int
    Returns:
    nearest neighbors: (batch_size, k, num_points)
    """
    dist = dist.contiguous()
    n = dist.size(1)
    if k > n:
        k = n
    top_dist, nn_idx = dist.topk(k=k,dim=1,largest=False)
    if ret_dist:
        return top_dist,nn_idx
    else:
        return nn_idx

def gather_nd(inputs, nn_idx):
    """
    input: (batch_size, num_dim, num_points)
    nn_idx:(batch_size, k, num_points)
    output:
    output:(batch_size, num_dim, k, num_points)
    """
    b, c, _ = inputs.size()
    _, k, n = nn_idx.size()

    # (b, c, k*n)
    nn_idx = nn_idx.unsqueeze(dim=1).expand(-1, c, -1, -1).view(b, c, -1)
    inputs_gather = inputs.gather(dim=-1, index=nn_idx)
    inputs_gather = inputs_gather.view(b, c, k, n)
    return inputs_gather

def get_inverse_density(pc_pos,k,sigma):
    # get density of every point
    pairwise_dist= pairwise_distance(pc_pos) # B x N x N
    nn_idx= knn(pairwise_dist, k)
    nn_pos= gather_nd(pc_pos, nn_idx)
    lnn_pos= nn_pos-pc_pos.unsqueeze(dim=2)
    density = kernel_density_estimation(lnn_pos,sigma,False) # density of all input points - B x 1 x N
    one = pc_pos.new(1)
    one[0]=1.0
    inverse_density = one.div(density)
    return inverse_density

def decay_weights(dist,sigma,tv_norm=False):
    '''
    dist - B x k x N
    return
    weights - B x k x N
    '''
    sigma_=dist.new(1).float()
    sigma_[0]=sigma
    sigma=sigma_.clone()
    weights=dist.div(sigma).pow(2).mul(-0.5).exp()
    if tv_norm:
        w_sum = weights.sum(1).mean(1)
    else:
        w_sum = weights.sum(dim=1,keepdim=True)
    weights/=w_sum
    return weights

