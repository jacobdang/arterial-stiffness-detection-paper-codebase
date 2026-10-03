from blackCrop_method1 import change_border_color_to_white as cb2w
import glob
import os
import cv2 as cv
import numpy as np
import sys
from PIL import ImageFile
import albumentations
from argparse import ArgumentParser
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
Ugly coded background cropping script
'''
def process_without_gb(img_raw):
    
    '''
    Script to crop the ROI from a Fundus image.
    Will use two different methods to get two results. You can pick the better one later when the product is online.
    
    Input:
            img_raw: the numpy array indicating the raw image
    
    Returns:
            result_img: a numpy array representing the cropped image
    '''
    
    h_raw,w_raw,c_raw = img_raw.shape
    new_h = 300
    new_w = int(300*(w_raw/h_raw))

    img = cv.resize(img_raw.copy(), (new_w, new_h), interpolation=cv.INTER_LINEAR)
    h,w,c = img.shape

    corner_pixel1 = (img[-1][-1][0]+img[-1][-1][1]+img[-1][-1][2])//3
    corner_pixel2 = (img[0][0][0]+img[0][0][1]+img[0][0][2])//3
    corner_pixel3 = (img[-1][0][0]+img[-1][0][1]+img[-1][0][2])//3
    corner_pixel4 = (img[0][-1][0]+img[0][-1][1]+img[0][-1][2])//3
    
    corner_pixel = (corner_pixel1+corner_pixel2+corner_pixel3+corner_pixel4)//4
    
    img_zero = corner_pixel*np.ones((h+200,w+200,c), dtype=img.dtype)
    img_zero[100:-100,100:-100,:] = img
    
    r_img, blob = cb2w(img_zero.copy()) 
    r_img = r_img[100:-100,100:-100,:]
    blob = blob[100:-100,100:-100]
    blob[blob>0] = 1
    blob = cv.resize(blob, (w_raw, h_raw), interpolation=cv.INTER_NEAREST)
    
    result_img = img_raw * blob[:,:,np.newaxis]
    result_img = img_raw[np.ix_(blob.any(1), blob.any(0))]
    
    max_dim = max(result_img.shape[0], result_img.shape[1])
    padder = albumentations.augmentations.transforms.PadIfNeeded(
            min_height=max_dim,min_width=max_dim,
            border_mode=0,always_apply=True)
    result_img = padder(image=result_img)["image"]

    return result_img