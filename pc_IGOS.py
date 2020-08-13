import os
import sys 
import glob
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import os.path as osp
import argparse

from open3d import utility,visualization,geometry

from modelnetdataset import *
from pointconv_utils import *
from pointconv import *
from blur_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

morph=True # whether to evaluate on curvature smoothing or on point del/ins
ins_mask=True # use l_ins
l11=0.3 # default l1 lambda
tvv=0.0 # default tv lambda
sigma=0.1 # for weight decay in upsampling
step_initial=200.0 # for line search of the step size
mmsize=256 # mask size

ckpt_path = 'log/model40_ap.ckpt'
            
def apply_plane_mask(pc,blurs,mask):
    '''
    pc - 1 x 3 x N
    blurs - 1 x l x 3 x N
    mask - 1 x 1 x N
    '''
    n=pc.shape[2]
    blurs=torch.cat([pc.unsqueeze(0),blurs],dim=1)
    l=blurs.shape[1]
    mat = pc.new(l).zero_() # l
    for i in range(l):
        mat[i]=i/10.0
    mat=mat.unsqueeze(0).unsqueeze(2).expand(-1,-1,n) # 1 x l x n
    mask = mask.expand(-1,l,-1) # 1 x l x n
    weights=((mask-mat)/0.05).pow(2).mul(-0.5).exp()
    weights=weights.unsqueeze(2).expand(-1,-1,3,-1) # 1 x l x 3 x N
    blurred_pc=blurs.mul(weights).sum(dim=1).div(weights.sum(dim=1)) # 1 x 3 x N
    return blurred_pc

def upsample_feature(pc_pos,sampled_pos,feature,k=5):
    '''
    upsample the feature with fewer points than pc_pos using linear interpolation
    pc_pos - B x 3 x N
    sampled_pos - pos of sampled points, B x 3 x n
    feature - B x c x n
    k - number of points in neighborhood

    return feature' - B x c x N
    '''
    B=pc_pos.shape[0]
    N=pc_pos.shape[2]
    c=feature.shape[1]
    n=sampled_pos.shape[2]
    if n>=N:
        #print("no need to upsample!")
        return feature
    if k>n:
        k=n
    pairwise_dist = pairwise_distance_general(pc_pos, sampled_pos) # B x n x N
    nn_dist,nn_idx = knn(pairwise_dist, k=k, ret_dist=True) # nearest sample dist and index - B x K x N, B x K x N
    nn_weights=decay_weights(nn_dist,sigma) # B x K x N, weights of the samples
    nn_weights_fat = nn_weights.unsqueeze(1).expand(-1,c,-1,-1)
    nn_features = gather_nd(feature, nn_idx) # B x c x K x N, features of the neighbor samples
    up_features = nn_features.mul(nn_weights_fat).sum(dim=2) # B x c x N 

    return up_features

def tv_norm(pc_pos, feature, tv_beta,k=4):
    '''
    get tv_norm of a feature
    pc_pos - B x 3 x N
    feature - B x c x N
    tv_beta - a constant
    k - number of points in neighborhood
    '''
    pairwise_dist = pairwise_distance(pc_pos) # B x N x N
    weights = decay_weights(pairwise_dist,sigma,tv_norm=True) # B x N x N
    feature_dist = pairwise_distance(feature) # B x N x N
    norm = (feature_dist*weights).sum()
    return norm

def l1_norm(feature):
    '''
    feature - B x c x N
    '''
    norm = feature.mean()
    return norm

def integrate_mask(pc_pos,model,target=None,k=10,max_iterations=30,integ_iter=20,tv_beta=2,
        l1_coeff=l11,tv_coeff=tvv,mask_size=mmsize,blurs=None,
        step_init=step_initial,step_low=0.00001,
        decay=0.2,beta=0.0001):
    '''
    Get the PC-IGOS mask for pc and model using pos as feature
    pc_pos - 1 x 3 x N
    model - a trained point conv classification model
    target - the label class, a long tensor, 1 x 1
    k - kernel size
    max_iterations - max iterations for deriving the mask
    integ_iter - number of iterations for approximating the integration of gradient
    tv_beta - a constant
    mask_size - size of mask, might be smaller than N
    blurs - the 10 precomputed smoothed shapes - 1 x 10 x 3 x N
    the rest are for backtracking line search for the step size
    '''
    import math
    B=pc_pos.shape[0]
    N=pc_pos.shape[2]
    if mask_size>N:
        mask_size=N

    score=F.softmax(model(pc_pos),dim=-1) # B x num_cls
    predict=torch.argmax(score,dim=-1,keepdim=True) # B x 1

    if target is not None:
        if target[0] != predict[0]:
            #print("Model failed to categorize correctly!")
            return None,None,None,None,None
    else:
        target=predict

    blurred_pos=blurs # 1 x 10 x 3 x N

    start=time.time()
    mask=pc_pos.new(B,1,mask_size).zero_()# B x 1 x n
    mask.requires_grad_()

    if mask_size==N:
        sampled_pos = pc_pos
    else:
        sampled_idx = fps_sampling(pc_pos, mask_size, True) # get indices of sampled points - B x n
        sampled_idx = sampled_idx.unsqueeze(dim=1).expand(-1,3,-1) # B x 3 x n
        sampled_pos = pc_pos.gather(dim=2, index=sampled_idx) # B x 3 x n

    optimizer = torch.optim.Adam([mask], lr=0.1)

    i=0

    while i < max_iterations:
        upsampled_mask = upsample_feature(pc_pos,sampled_pos,mask,k=k) # B x 1 x N
        l1_loss = l1_coeff*l1_norm(upsampled_mask)
        tv_loss = tv_coeff*tv_norm(pc_pos,upsampled_mask,tv_beta)
        loss1 = l1_loss+tv_loss
        loss_all = loss1.clone()

        masked_pc_del_base=apply_plane_mask(pc_pos,blurred_pos,upsampled_mask)
        loss_del_base=F.softmax(model(masked_pc_del_base),dim=-1)[:,target] # B
        loss_base = loss1 + loss_del_base
        if ins_mask:
            masked_pc_ins_base=apply_plane_mask(pc_pos,blurred_pos,1-upsampled_mask)
            loss_ins_base=-F.softmax(model(masked_pc_ins_base),dim=-1)[:,target] # B

            loss_base += loss_ins_base
        #print("loss_del_base",loss_del_base)

        for inte_i in range(integ_iter):
            integ_mask = upsampled_mask+((inte_i+1.0)/integ_iter)*(1.0-upsampled_mask)
            masked_pc_del=apply_plane_mask(pc_pos,blurred_pos,integ_mask)
            noise=torch.randn_like(masked_pc_del)/50.0
            masked_pc_del=(masked_pc_del+noise)
            loss_del=F.softmax(model(masked_pc_del),dim=-1)[:,target] 
            loss_all = loss_all + loss_del/integ_iter
            if ins_mask:
                masked_pc_ins=apply_plane_mask(pc_pos,blurred_pos,1-integ_mask)
                masked_pc_ins=(masked_pc_ins+noise)
                loss_ins=-F.softmax(model(masked_pc_ins),dim=-1)[:,target]
                loss_all += loss_ins/integ_iter
        optimizer.zero_grad()
        mask.retain_grad()
        loss_all.backward(retain_graph=True)
        whole_grad = mask.grad.data.clone()

        # line search
        step = step_init
        mask_stepped = mask-step*whole_grad
        mask_stepped.clamp_(0,1)
        mask_stepped_up = upsample_feature(pc_pos,sampled_pos,mask_stepped,k=k)
        masked_pc_del_stepped = apply_plane_mask(pc_pos,blurred_pos,mask_stepped_up)
        loss_del_stepped=F.softmax(model(masked_pc_del_stepped),dim=-1)[:,target]
        loss_stepped=l1_coeff*l1_norm(mask_stepped_up)+tv_coeff*tv_norm(pc_pos,mask_stepped_up,tv_beta)+loss_del_stepped
        if ins_mask:
            masked_pc_ins_stepped = apply_plane_mask(pc_pos,blurred_pos,1-mask_stepped_up)
            loss_ins_stepped=-F.softmax(model(masked_pc_ins_stepped),dim=-1)[:,target]
            loss_stepped+=loss_ins_stepped
        new_condition = whole_grad ** 2  # Here the direction is the whole_grad
        new_condition = new_condition.sum() # Assume B = 1
        new_condition = beta * step * new_condition

        while loss_stepped > loss_base - new_condition:
            step *= decay
            if step<step_low:
                break
            mask_stepped = mask-step*whole_grad
            mask_stepped.clamp_(0,1)

            mask_stepped_up = upsample_feature(pc_pos,sampled_pos,mask_stepped,k=k)
            masked_pc_del_stepped = apply_plane_mask(pc_pos,blurred_pos,mask_stepped_up)
            loss_del_stepped=F.softmax(model(masked_pc_del_stepped),dim=-1)[:,target]
            loss_stepped=l1_coeff*l1_norm(mask_stepped_up)+tv_coeff*tv_norm(pc_pos,mask_stepped_up,tv_beta)+loss_del_stepped
            if ins_mask:
                masked_pc_ins_stepped = apply_plane_mask(pc_pos,blurred_pos,1-mask_stepped_up)
                loss_ins_stepped=-F.softmax(model(masked_pc_ins_stepped),dim=-1)[:,target]
                loss_stepped+=loss_ins_stepped
            new_condition = whole_grad ** 2  # Here the direction is the whole_grad
            new_condition = new_condition.sum() # Assume B = 1
            new_condition = beta * step * new_condition

        mask = mask.clone() - step * whole_grad
        mask.clamp_(0,1)

        del loss_del_stepped
        del loss_del_base, loss_base
        del loss_all
        if ins_mask:
            del loss_ins_stepped
            del loss_ins_base
        i+=1

    upsampled_mask = upsample_feature(pc_pos,sampled_pos,mask,k=k) # B x 1 x N

    end=time.time()
    return mask, sampled_pos, upsampled_mask, blurred_pos, l1_coeff 

def evaluate_on_all_classes(log=True,overall=False,l1=l11,tv=tvv,visualize=False):
    '''
    evaluate PC-IGOS on the ModelNet40 test split
    log - print the curves of each class
    overall - print the result of the entire split
    visualize - save the point clouds along the del/ins curve and the mask for future visualization
                turn it off during actual evaluation
    '''
    num_points=1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pcnet=SpiderPC(3,40,num_points,fix_fps=True).to(device)
    pcnet.eval()
    # load status from checkpoint
    if os.path.isfile(ckpt_path):
        pcnet.load_state_dict(torch.load(ckpt_path,map_location=device))
        print('Loaded ckpt!')
    class_count=0.0 # how many classes are evaluated so far
    deletion_overall=0.0 # summed AUC of the deletion curve
    insertion_overall=0.0 # summed AUC of the insertion curve
    diff_overall=0.0
    for shape in range(40):
        if log:
            print("class:",shape)
        if overall:
            print("#",end="",flush=True)
        dataset=Model40DataSet(model10=False, train=False, normal=False,augmentation=False,sample_num=num_points,shape=shape,load_blur=True)
        dataloader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=1)
        count=0.0 # count within the class
        score_orig=0.0
        score_deleted=torch.zeros(21).to(device) # record for every five percent
        score_deleted_reverse=torch.zeros(21).to(device) # record for every five percent
        for i,(pc_batch,targets,blurs) in enumerate(dataloader):
            if log:
                print("#",end="",flush=True)
            n=pc_batch.shape[2]
            pc_batch, targets = pc_batch.to(device), targets.to(device)
            pc_pos=pc_batch[:,:3,:]
            blurs=blurs.to(device)
            mask,sampled_pos,mask_upsampled,blurred_pos,l1_coeff=integrate_mask(pc_pos,pcnet,targets,l1_coeff=l1,tv_coeff=tv,blurs=blurs)
            if mask is None or mask.abs().sum()<0.00000001:
                # failure case. if want to save the results, increase idx for synchronization.
                # otherwise does not count this one.
                if visualize:
                    count+=1
                continue
            mask_size=mask.shape[2]
            out_origin=F.softmax(pcnet(pc_pos),dim=1)[0,targets[0]]
            deletion_mask=mask.new(mask.shape).zero_()
            deletion_mask_reverse=mask.new(mask.shape).zero_()

            tmp_deletes=list() # store the gradually morphed shapes along del curve
            tmp_inserts=list() # store the gradually morphed shapes along ins curve
            tmp_scores=list()
            tmp_insert_scores=list()
            for j in range(21):
                if not morph and j==20:
                    # if use point del/ins, j==20 means there is no point to delete
                    continue
                if morph:
                    _,topidx=torch.topk(mask,round((mask_size/20)*(j)),dim=-1,largest=True)
                    deletion_mask.scatter_(dim=-1,index=topidx,value=1.0)
                    deletion_mask_up=upsample_feature(pc_pos,sampled_pos,deletion_mask,k=10)
                    deleted_pc=apply_plane_mask(pc_pos,blurred_pos,deletion_mask_up)
                else:
                    _,keepidx=torch.topk(mask_upsampled,num_points-round((num_points/20)*(j)),dim=-1,largest=False)
                    deleted_pc=pc_pos.gather(dim=2,index=keepidx.expand(-1,3,-1))
                if j==0:
                    deleted_pc=pc_pos
                out_deleted=F.softmax(pcnet(deleted_pc),dim=1)[0,targets[0]]
                if visualize:
                    tmp_deletes.append(deleted_pc[0].clone())
                    tmp_scores.append(out_deleted.clone().item())
                score_deleted[j]+=out_deleted.item()
                del out_deleted
            
                # delete points with smallest values
                if morph:
                    _,topidx=torch.topk(mask,round((mask_size/20)*(j)),dim=-1,largest=False)
                    deletion_mask_reverse.scatter_(dim=-1,index=topidx,value=1.0)
                    deletion_mask_reverse_up=upsample_feature(pc_pos,sampled_pos,deletion_mask_reverse,k=10)
                    deleted_pc=apply_plane_mask(pc_pos,blurred_pos,deletion_mask_reverse_up)
                else:
                    _,keepidx=torch.topk(mask_upsampled,num_points-round((num_points/20)*(j)),dim=-1,largest=True)
                    deleted_pc=pc_pos.gather(dim=2,index=keepidx.expand(-1,3,-1))
                if j==0:
                    deleted_pc=pc_pos
                out_deleted=F.softmax(pcnet(deleted_pc),dim=1)[0,targets[0]]
                if visualize:
                    tmp_inserts.append(deleted_pc[0].clone())
                    tmp_insert_scores.append(out_deleted.clone().item())
                score_deleted_reverse[j]+=out_deleted.item()
                del out_deleted

            if visualize:
                masked_pc=apply_plane_mask(pc_pos,blurred_pos,mask_upsampled)
                out_masked=F.softmax(pcnet(masked_pc),dim=1)[0,targets[0]]
                if tmp_scores[20]<=0.2*tmp_scores[0]: # only save the results if the del curve is valid
                    shape_name=dataset.shape_list[shape]
                    torch.save(masked_pc[0],'tensors/'+shape_name+"-"+str(int(count))+'-{:.2f}'.format(out_masked)+'-masked_pc.pt')
                    torch.save(mask_upsampled[0],'tensors/'+shape_name+"-"+str(int(count))+'-color_mask.pt')
                    for j in range(11):
                        tmp_del=tmp_deletes[j*2].unsqueeze(0)
                        tmp_ins=tmp_inserts[20-j*2].unsqueeze(0)
                        score=F.softmax(pcnet(tmp_del),dim=-1) # B x num_cls
                        predict=torch.argmax(score,dim=-1,keepdim=True) # B x 1
                        predict_shape=dataset.shape_list[predict]
                        torch.save(tmp_deletes[j*2], 'tensors/'+shape_name+"-del-"+str(int(count))+"-"+str(j*10)+"-"+"{:.2f}".format(tmp_scores[j*2])+'-'+predict_shape+'-'+"{:.2f}".format(score.max())+'.pt')
                        score=F.softmax(pcnet(tmp_ins),dim=-1) # B x num_cls
                        predict=torch.argmax(score,dim=-1,keepdim=True) # B x 1
                        predict_shape=dataset.shape_list[predict]
                        torch.save(tmp_inserts[20-j*2], 'tensors/'+shape_name+"-ins-"+str(int(count))+"-"+str(j*10)+"-"+"{:.2f}".format(tmp_insert_scores[20-2*j])+'-'+predict_shape+'-'+"{:.2f}".format(score.max())+'.pt')
                    print(shape_name+" saved pcs")

            count+=1.0
            score_orig+=out_origin.item()

        if log:
            print("")
        if count == 0.0:
            if log:
                print("Class {} does not have any count!".format(shape))
                print("")
            continue
        else:
            class_count+=1.0
        score_orig/=count
        score_deleted/=count
        score_deleted_reverse/=count

        if log:
            print("Class: {}, avg original score: {:.4f}, avg deletion sum: {:.4f}, avg reverse deletion sum: {:.4f}, count: {}"
                    .format(dataset.shape_list[shape],score_orig,score_deleted.mean().item(),score_deleted_reverse.mean().item(),int(count)))
            print("Avg deletion score from 0% to 100%: ")
            for j in range(21):
                print("{:.2f}".format(score_deleted[j].item()),end=" ",flush=True)
            print("")
            print("Avg reverse deletion score from 0% to 100%: ")
            for j in range(21):
                print("{:.2f}".format(score_deleted_reverse[j].item()),end=" ",flush=True)
            print("")
            print("")
        if overall:
            deletion_overall+=score_deleted.mean().item()
            insertion_overall+=score_deleted_reverse.mean().item()
            diff_overall+=(score_deleted_reverse-score_deleted).mean().item()
    if overall:
        print("")
        print("deletion_overall:",deletion_overall/class_count)
        print("insertion_overall:",insertion_overall/class_count)
        print("diff_overall:",diff_overall/class_count)
        print("")

def grid_search():
    tv=[0,0.0001]
    l1=[0.003,0.3]
    for i in range(len(tv)):
        for j in range(len(l1)):
            print("tv:",str(tv[i]))
            print("l1:",str(l1[j]))
            evaluate_on_all_classes(log=False,overall=True,l1=l1[j],tv=tv[i])

if __name__ == '__main__':
    evaluate_on_all_classes()
    #grid_search()
