import os
import sys 
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointconv_utils import *

def surface_neighbor(pc,k=4,r=2,d=0):
    '''
    Get surface neighborhood for each point in pc
    pc - B x 3 x N
    k - integer, number of points in each layer
    r - integer, number of layer
    d - integer, number of dilation 
    '''
    b,_,n=pc.size()
    #pairwise_dist = pairwise_distance(pc).exp() # B x N x N
    pairwise_dist = pairwise_distance(pc) # B x N x N
    nn_idx = knn(pairwise_dist, k=k+d) # B x k x N
    nn_idx = torch.cat([nn_idx[:,0:1,:],nn_idx[:,d+1:,:]],dim=1)
    layer_idx_all=nn_idx
    for i in range(r-1):
        l=layer_idx_all.shape[1]
        layer_idx_long=layer_idx_all.view(b,-1).unsqueeze(1).expand(-1,k,-1) # B x k x (l*N)
        layer_idx=nn_idx.gather(dim=2,index=layer_idx_long) # B x k x (l*N)
        layer_idx=layer_idx.view(b,k*l,n) # enlarged neighborhood
        layer_idx_all=torch.cat([layer_idx_all,layer_idx],dim=1)
    return layer_idx_all

def test_surface_neighbor():
    dataset=Model40DataSet(True,True,False)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=1)
    for i,(pc_batch,targets) in enumerate(dataloader):
        pc=pc_batch[:,:3,:] # 1 x 3 x N
        n=pc.shape[2]
        nn_idx=surface_neighbor(pc,k=5,r=2) # 1 x l x N
        nn_idx_batch=nn_idx.split(1,dim=2)
        for nn in nn_idx_batch:
            nn=nn[:,:,0] # 1 x l
            mask=nn.new(1,n).zero_().float() # 1 x n
            mask.scatter_(dim=-1,index=nn,value=1.0)
            visualize_pc(pc[0],mask)

def get_fitted_line(nn_pos):
    '''
    nn_pos - 2 x l, tensor
    '''
    x=nn_pos.new(nn_pos.shape).zero_() # 2 x l
    x[0]=nn_pos[0]
    x[1,:]=1.0
    y=nn_pos[1,:].unsqueeze(1) # l x 1
    try:
        sol,_=torch.lstsq(y,x.permute(1,0))[:2] # a,c
    except RuntimeError:
        sol1=nn_pos.new(3).zero_()
        sol1[0]=0
        sol1[1]=1
        sol1[2]=0
    else:
        sol1=nn_pos.new(3).zero_()
        sol1[0]=sol[0]
        sol1[1]=-1
        sol1[2]=sol[1]

    x=nn_pos.new(nn_pos.shape).zero_() # 2 x l
    x[0]=nn_pos[1]
    x[1,:]=1.0
    y=nn_pos[0,:].unsqueeze(1) # l x 1
    try:
        sol,_=torch.lstsq(y,x.permute(1,0))[:2] # b,c
    except RuntimeError:
        sol2=nn_pos.new(3).zero_()
        sol2[0]=1
        sol2[1]=0
        sol2[2]=0
    else:
        sol2=nn_pos.new(3).zero_()
        sol2[0]=-1
        sol2[1]=sol[0]
        sol2[2]=sol[1]

    normal=sol1[:2].unsqueeze(1)
    norm=normal.norm()
    loss=nn_pos.mul(normal).sum(dim=0)
    loss=(loss+sol1[2])/norm
    loss=loss.pow(2) # l
    loss=loss.sum()
    loss1=loss.clone()

    normal=sol2[:2].unsqueeze(1)
    norm=normal.norm()
    loss=nn_pos.mul(normal).sum(dim=0)
    loss=(loss+sol2[2])/norm
    loss=loss.pow(2) # l
    loss=loss.sum()
    loss2=loss.clone()

    if loss1>loss2:
        return sol2
    else:
        return sol1

def get_fitted_plane(nn_pos):
    '''
    nn_pos - 3 x l, torch tensor 
    '''
    x=nn_pos.new(3,nn_pos.shape[1]).zero_()
    x[:2,:]=nn_pos[:2,:] # 3 x l
    x[2,:]=1.0
    y=nn_pos[2,:].unsqueeze(1) # l x 1
    #start=time.time()
    try:
        sol,_=torch.lstsq(y,x.permute(1,0))[:3] # a,b,d
    except RuntimeError:
        sol1=nn_pos.new(4).zero_()
        sol1[0]=0
        sol1[1]=0
        sol1[2]=1
        sol1[3]=0
    else:
        #end=time.time()
        #print("lstsq:",end-start)
        sol1=nn_pos.new(4).zero_()
        sol1[0]=sol[0]
        sol1[1]=sol[1]
        sol1[2]=-1
        sol1[3]=sol[2]

    x=nn_pos.new(3,nn_pos.shape[1]).zero_()
    x[:2,:]=nn_pos[1:,:]# 3 x l
    x[2,:]=1.0
    y=nn_pos[0,:].unsqueeze(1) # l x 1
    try:
        sol,_=torch.lstsq(y,x.permute(1,0))[:3] # b,c,d
    except RuntimeError:
        sol2=nn_pos.new(4).zero_()
        sol2[0]=1
        sol2[1]=0
        sol2[2]=0
        sol2[3]=0
    else:
        sol2=nn_pos.new(4).zero_()
        sol2[0]=-1
        sol2[1]=sol[0]
        sol2[2]=sol[1]
        sol2[3]=sol[2]

    x=nn_pos.new(3,nn_pos.shape[1]).zero_()
    x[:2,:]=torch.cat([nn_pos[0:1,:],nn_pos[2:,:]],dim=0)# 3 x l
    x[2,:]=1.0
    y=nn_pos[1,:].unsqueeze(1) # l x 1
    try:
        sol,_=torch.lstsq(y,x.permute(1,0))[:3] # a,c,d
    except RuntimeError:
        sol3=nn_pos.new(4).zero_()
        sol3[0]=0
        sol3[1]=1
        sol3[2]=0
        sol3[3]=0
    else:
        sol3=nn_pos.new(4).zero_()
        sol3[0]=sol[0]
        sol3[1]=-1
        sol3[2]=sol[1]
        sol3[3]=sol[2]

    normal=sol1[:3].unsqueeze(1)
    norm=normal.norm()
    loss=nn_pos.mul(normal).sum(dim=0)
    loss=(loss+sol1[3])/norm
    loss=loss.pow(2) # l
    loss=loss.sum()
    loss1=loss.clone()

    normal=sol2[:3].unsqueeze(1)
    norm=normal.norm()
    loss=nn_pos.mul(normal).sum(dim=0)
    loss=(loss+sol2[3])/norm
    loss=loss.pow(2) # l
    loss=loss.sum()
    loss2=loss.clone()

    normal=sol3[:3].unsqueeze(1)
    norm=normal.norm()
    loss=nn_pos.mul(normal).sum(dim=0)
    loss=(loss+sol3[3])/norm
    loss=loss.pow(2) # l
    loss=loss.sum()
    loss3=loss.clone()

    if loss1>loss2:
        if loss2 > loss3:
            return sol3
        else:
            return sol2
    else:
        if loss1 > loss3:
            return sol3
        else:
            return sol1

def project_to_line(nn_pos,x,normal,plane,erosion,lamb,miu):
    # project all neighbors to the plane
    dist=x.dot(normal)+plane[3]
    x_proj=x-normal*dist
    nn_pos_dist=normal.unsqueeze(0).mm(nn_pos)+plane[3] # 1 x l
    nn_pos_proj=nn_pos-nn_pos_dist.expand(3,-1).mul(normal.unsqueeze(1)) # 3 x l
    # get u,v direction on the plane
    uvec=nn_pos_proj[:,0]-x_proj # 3
    uvec=uvec/torch.norm(uvec)
    vvec=torch.cross(normal,uvec)
    vvec=vvec/torch.norm(vvec)
    # get u,v coordinates of all points, x is just (0,0)
    uv=uvec.new(2,2).zero_() # first two coords are enough
    uv[:,0]=uvec[:2]
    uv[:,1]=vvec[:2]
    try:
        uv_inverse=torch.inverse(uv) # 2 x 2
    except RuntimeError:
        uv[0,0]=uvec[0]
        uv[1,0]=uvec[2]
        uv[0,1]=vvec[0]
        uv[1,1]=vvec[2]
        uv_inverse=torch.inverse(uv) # 2 x 2
        third=True
    else:
        third=False
    x_uv=uv.new(2).zero_() # 2
    if third:
        nn_trans=nn_pos_proj-x_proj.unsqueeze(1) # 3 x l
        nn_short=nn_trans.new(2,nn_trans.shape[1]).zero_()
        nn_short[0]=nn_trans[0]
        nn_short[1]=nn_trans[2]
    else:
        nn_short=(nn_pos_proj-x_proj.unsqueeze(1))[:2]
    nn_pos_uv=uv_inverse.mm(nn_short) # 2 x l
    # get the fitted line
    line=get_fitted_line(nn_pos_uv) # a, b, c
    lnorm=torch.norm(line[:2])
    line=line/lnorm
    lnormal=line[:2]
    ldist=x_uv.dot(lnormal)+line[2]
    if erosion:
        x_uv_proj=x_uv-lnormal*(ldist*lamb)
    else:
        x_uv_proj=x_uv+lnormal*(ldist*miu)
    x_proj_2=x+uvec*x_uv_proj[0]+vvec*x_uv_proj[1] # 3
    return x_proj_2

def plane_smoothing(pc,k=4,r=2,d=0,lamb=0.9,miu=1.8,erosion=True,subset=None):
    '''
    pc - 1 x 3 x N
    subset - a list of idx
    k - number of neighbor
    r - number of layers
    d - dilation (for nearest neighbor)
    ma - maximum dilation times (not the same dilation as above)
    '''
    pc_orig=pc.clone()
    pc=pc.clone()
    pc_copy=pc[0].clone()
    n=pc.shape[2]
    #start=time.time()
    nn_idx=surface_neighbor(pc,k=k,r=r,d=d) # 1 x l x N
    #end=time.time()
    nn_idx_batch=nn_idx[0].split(1,dim=1)
    if subset is None:
        subset=range(n)
    count=0.0
    for i in subset:
        idx=nn_idx_batch[i][:,0] # l
        idx=idx.index_select(dim=0,index=(idx-i).nonzero().squeeze())
        idx=idx.unique()
        nn_pos=pc_copy.index_select(dim=1,index=idx) # 3 x l
        x=pc_copy[:,i] # 3
        plane=get_fitted_plane(nn_pos) #a,b,c,d
        norm=torch.norm(plane[:3])
        plane=plane/norm
        normal=plane[:3]
        dist=x.dot(normal)+plane[3]
        x=project_to_line(nn_pos,x,normal,plane,erosion,lamb,miu)
        if erosion:
            projected=x-normal*(dist*lamb) # 3
        else: # dilation
            projected=x+normal*(dist*miu) # 3
        pc[0,:,i]=projected
    return pc

def plane_blur(pc,mask=None,single=False):
    '''
    pc - 1 x 3 x N
    mask - 1 x 1 x N
    '''
    pcs=pc.new(1,10,3,pc.shape[2])
    pc=pc.clone().cpu()
    pc_orig=pc.clone()
    x_orig=pc[0][0].abs().mean()
    y_orig=pc[0][1].abs().mean()
    z_orig=pc[0][2].abs().mean()
    mean_orig=pc.abs().mean()
    idx = None
    if mask is not None:
        idx=mask[0][0].ge(0.5).long().nonzero()
        idx=idx.squeeze()
    ks=[20,30,40,50,60]
    rs=[2,2,2,2,2]
    for j in range(2*len(ks)):
        k=ks[j//2]
        r=rs[j//2]
        lamb=0.7
        miu=1.0
        for z in range(8):
            pc=plane_smoothing(pc,k=k,r=r,d=0,lamb=lamb,miu=miu,erosion=True,subset=idx)
            pc=plane_smoothing(pc,k=k,r=r,d=0,lamb=lamb,miu=miu,erosion=False,subset=idx)
            if torch.isnan(pc).sum() != 0.0:
                print("intel error!")
                return None
        pc_new=pc.clone()
        mean_cur=pc.abs().mean()
        pc_new[0]=pc[0]*mean_orig/mean_cur
        pc_new=pc_new-pc_new.mean(2,keepdim=True)
        pc_new=pc_new*pow(0.97,j)
        pcs[:,j,:,:]=pc_new.clone()
    if single:
        return pcs[:,-1,:,:]
    else:
        return pcs


'''
quadric
'''
def quadric(x,y,z,a,b,c,d,e,f,g,h,i,j): 
    #fit quadric surface
    ret=a*x**2+b*y**2+c*z**2+d*x*y+e*y*z+f*x*z+g*x+h*y+i*z+j
    return ret

def residual(params, points):
    #total residual
    residuals = [
      quadric(p[0], p[1], p[2],
      params[0], params[1], params[2], params[3], params[4],
      params[5], params[6], params[7], params[8], params[9]) for p in points]
    return np.linalg.norm(residuals)

def get_fitted_quadric(poi,nn_pos):
    total_pos=torch.cat([poi.unsqueeze(1),nn_pos],dim=1) #3 x n
    x=total_pos[0] # n
    y=total_pos[1]
    z=total_pos[2]
    p=total_pos.new(10,x.shape[0]) # 10 x n
    p[0]=x.pow(2)
    p[1]=y.pow(2)
    p[2]=z.pow(2)
    p[3]=x*y
    p[4]=y*z
    p[5]=x*z
    p[6]=x
    p[7]=y
    p[8]=z
    p[9]=1
    R=p.matmul(p.transpose(0,1)) # 10x10
    C=R[:6,:6] #6x6
    B=R[:6,6:] #6x4
    A=R[6:,6:] #4x4
    two=A.new(1).zero_()
    two[0]=2
    sqrt_two=two.pow(0.5)
    H=A.new(6,6).zero_() #6x6
    H[0][0]=1
    H[1][1]=1
    H[2][2]=1
    H[3][3]=1/sqrt_two
    H[4][4]=1/sqrt_two
    H[5][5]=1/sqrt_two
    try:
        M=C-B.matmul(A.inverse().matmul(B.transpose(0,1)))
    except RuntimeError:
        A+=0.01
        M=C-B.matmul(A.inverse().matmul(B.transpose(0,1)))
    M_prime=H.inverse().matmul(M.inverse()).matmul(H.inverse())
    evals,evecs=M_prime.eig(eigenvectors=True) #6x2, 6x6, column is evec
    evals=(evals[:,0].pow(2)+evals[:,1].pow(2)).pow(0.5) # 6
    min_index=evals.argmin()
    beta_prime=evecs[:,min_index] # 6
    beta=H.inverse().matmul(beta_prime)
    alpha=-A.inverse().matmul(B.transpose(0,1)).matmul(beta) #4
    result=alpha.new(10).zero_()
    result[:6]=beta
    result[6:]=alpha
    return result

def get_gradient(p,quadric):
    a=quadric[0]
    b=quadric[1]
    c=quadric[2]
    d=quadric[3]
    e=quadric[4]
    f=quadric[5]
    g=quadric[6]
    h=quadric[7]
    i=quadric[8]
    j=quadric[9]
    x=p[0]
    y=p[1]
    z=p[2]
    fx=2*a*x+d*y+f*z+g
    fy=2*b*y+d*x+e*z+h
    fz=2*c*z+e*y+f*x+i
    grad=p.new(3).zero_()
    grad[0]=fx
    grad[1]=fy
    grad[2]=fz
    return grad

def get_hessian(quadric):
    a=quadric[0]
    b=quadric[1]
    c=quadric[2]
    d=quadric[3]
    e=quadric[4]
    f=quadric[5]
    g=quadric[6]
    h=quadric[7]
    i=quadric[8]
    j=quadric[9]
    hess=quadric.new(3,3).zero_()
    hess[0][0]=2*a
    hess[0][1]=d
    hess[0][2]=f
    hess[1][0]=d
    hess[1][1]=2*b
    hess[1][2]=e
    hess[2][0]=f
    hess[2][1]=e
    hess[2][2]=2*c
    trace=2*(a+b+c)
    return hess,trace

def get_mean_curvature(grad,grad_norm,hess,trace):
    mc=grad.unsqueeze(0).matmul(hess.matmul(grad.unsqueeze(1)))-grad_norm.pow(2)*trace
    mc=mc/(2*grad_norm.pow(3))
    return mc

def quadric_smoothing(pc,k=4,r=2,d=0,lamb=0.9,miu=1.8,erosion=True,subset=None):
    '''
    pc - 1 x 3 x N
    subset - a list of idx
    k - number of neighbor
    r - number of layers
    d - dilation (for nearest neighbor)
    ma - maximum dilation times (not the same dilation as above)
    '''
    pc_orig=pc.clone()
    pc=pc.clone()
    pc_copy=pc[0].clone()
    n=pc.shape[2]
    nn_idx=surface_neighbor(pc,k=k,r=r,d=d) # 1 x l x N
    nn_idx_batch=nn_idx[0].split(1,dim=1)
    if subset is None:
        subset=range(n)
    count=0.0
    for i in subset:
        idx=nn_idx_batch[i][:,0] # l
        idx=idx.index_select(dim=0,index=(idx-i).nonzero().squeeze())
        idx=idx.unique()
        nn_pos=pc_copy.index_select(dim=1,index=idx) # 3 x l
        x=pc_copy[:,i] # 3
        quadric=get_fitted_quadric(x,nn_pos) #a,b,c,d,e,f,g,h,i,j

        #calculate normal unit vector
        gradient=get_gradient(x,quadric) # 3
        grad_norm=gradient.norm()
        normal=gradient/grad_norm
        hess,trace=get_hessian(quadric) # 3 x 3
        #calculate mean curvature
        curvature=get_mean_curvature(gradient,grad_norm,hess,trace) #a signed scalar
        dist=curvature
        #print(dist)
        if erosion:
            projected=x+normal*(dist*lamb) # 3
        else: # dilation
            projected=x-normal*(dist*miu) # 3
        pc[0,:,i]=projected
    return pc


def quadric_blur(pc,mask=None,single=False):
    '''
    pc - 1 x 3 x N
    mask - 1 x 1 x N
    '''
    from modelnetdataset import visualize_pc 
    pcs=pc.new(1,10,3,pc.shape[2])
    pc=pc.clone().cpu()
    pc_orig=pc.clone()
    x_orig=pc[0][0].abs().mean()
    y_orig=pc[0][1].abs().mean()
    z_orig=pc[0][2].abs().mean()
    mean_orig=pc.abs().mean()
    idx = None
    if mask is not None:
        idx=mask[0][0].ge(0.5).long().nonzero()
        idx=idx.squeeze()
        #print(idx)
    ks=[20,30,40,50,60]
    rs=[2,2,2,2,2]
    #pcs=list()
    for j in range(2*len(ks)):
        #print(j)
        #start=time.time()
        k=ks[j//2]
        r=rs[j//2]
        lamb=0.002
        miu=0.0022
        for z in range(2):
            pc=quadric_smoothing(pc,k=k,r=r,d=0,lamb=lamb,miu=miu,erosion=True,subset=idx)
            pc=quadric_smoothing(pc,k=k,r=r,d=0,lamb=lamb,miu=miu,erosion=False,subset=idx)
            if torch.isnan(pc).sum() != 0.0:
                print("intel error!")
                return None
        #end=time.time()
        #print("time on gpu:",end-start)
        pc_new=pc.clone()
        mean_cur=pc.abs().mean()
        pc_new[0]=pc[0]*mean_orig/mean_cur
        pc_new=pc_new-pc_new.mean(2,keepdim=True)
        pc_new=pc_new*pow(0.97,j)
        #visualize_pc(pc_new[0])
        pcs[:,j,:,:]=pc_new.clone()
    if single:
        return pcs[:,-1,:,:]
    else:
        return pcs


if __name__ == '__main__':
    #suppose you have a pc_pos of shape 1 x 3 x 1024
    blurs=plane_blur(pc_pos,mask=None,single=False) # pc_pos must be of shape 1 x 3 x N
    '''
    if mask is not None, only points with mask value >=0.5 will be blurred
    if single is False, output is 1 x 10 x 3 x N, 10 levels of blurriness, from weak to strong
    if single is True, output is 1 x 3 x N, only return the strongest blurred version (basically a sphere)
    '''
