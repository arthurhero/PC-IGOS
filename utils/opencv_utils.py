import cv2
import numpy as np
import random

import torch

def add_alpha(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2RGBA)
    return img

def flip_img_b(img):
    img = cv2.flip(img, -1)
    return img

def resize_img(img,x,y):
    img = cv2.resize(img,(x,y))
    return img

def flip_img_v(img):
    img = cv2.flip(img, 0)
    return img

def flip_img_h(img):
    img = cv2.flip(img, 1)
    return img

def load_img(fname):
    img = cv2.imread(fname,cv2.IMREAD_UNCHANGED)
    return img

def save_img(fname,img):
    cv2.imwrite(fname,img)

def display_img(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_torch_img(img):
    img=img.cpu().detach()
    img = (img+0.5).clamp(0,1.0)
    img = (img*255).long().permute(1,2,0).numpy()
    img=img.astype(np.uint8)
    display_img(img)

def display_img_file(fname):
    img = cv2.imread(fname)
    cv2.imshow(fname,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bbox_img(img,y1,x1,y2,x2,color=(255,0,0)):
    ret=img.copy()
    cv2.rectangle(ret,(x1,y1),(x2,y2),color,2)
    return ret

def generate_random_vertices(height=128,width=128,num_min=3,num_max=6,dis_max_ratio=0.5):
    '''
    generates an array of points for a polygon to be drawn on a square
    mask of size size.
    number of points restricted by num_min and num_max
    dis_max_ratio indicates the maximum distance between the coordinates of 
    two consecutive points w.r.t size
    '''
    dis_max = round(dis_max_ratio*min(height,width))
    num = random.randint(0,num_max-num_min+1)+num_min
    pts = list()
    x=random.randint(0,width)
    y=random.randint(0,height)
    pts.append([x,y])
    for i in range(num):
        x_off=random.randint(0,dis_max*2+1)-dis_max
        x+=x_off
        y_off=random.randint(0,dis_max*2+1)-dis_max
        y+=y_off
        pts.append([x,y])
    pts = np.asarray(pts, dtype=np.int32)
    return pts

def generate_polygon_mask(height=128,width=128,pts=None):
    '''
    generates a random polygon mask on canvas of size size
    0 for non-mask area and 1 for mask
    '''
    mask = np.zeros((height,width,1),np.float32)
    if pts is None:
        pts = generate_random_vertices(height,width)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(mask, [pts], 1)
    return mask
