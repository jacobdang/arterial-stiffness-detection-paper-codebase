from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.segmentation import flood
from skimage.measure import regionprops
from skimage.morphology import label, binary_opening, binary_closing, disk
from copy import deepcopy
import numpy as np
import os
import cv2
from skimage.feature import canny
from scipy import ndimage as ndi

def _get_center_by_edge(mask):

    mask_props = regionprops(mask)
    center= mask_props[0].centroid
    axis_length = mask_props[0].major_axis_length

    center = (mask.shape[0]//2, mask.shape[1]//2)
    
    return center, axis_length


def _get_radius_by_mask_center(mask,center,input_radius=None):

    if input_radius is not None:
        return input_radius

    mask=mask.astype(np.uint8)
    ksize=max(mask.shape[1]//400*2+1,3)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
    mask=cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    
    index=np.where(mask>0)
    d_int=np.sqrt((index[0]-center[0])**2+(index[1]-center[1])**2)
    radius = np.max(d_int).astype(int)
    
    return radius


def _get_circle_by_center_bbox(shape,center,bbox,radius):
    center_mask=np.zeros(shape=shape).astype('uint8')
    tmp_mask=np.zeros(shape=bbox[2:4])
    center_tmp=(int(center[0]),int(center[1]))
    center_mask=cv2.circle(center_mask,center_tmp[::-1],int(radius),(1),-1)
    # center_mask[bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3]]=tmp_mask
    # center_mask[bbox[0]:min(bbox[0]+bbox[2],center_mask.shape[0]),bbox[1]:min(bbox[1]+bbox[3],center_mask.shape[1])]=tmp_mask
    return center_mask

def change_border_color_to_white(img_rgb):
    img_gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)*1.5
    img_shape = img_gray.shape
    
    marker_set = ((5, 5), (5, img_shape[1]-5), (img_shape[0]-5, 5), (img_shape[0]-5, img_shape[1]-5))
    marker_color_set = []
    for marker_coords in marker_set:
        marker_color_set.append(img_gray[(marker_coords[0], marker_coords[1])])
    marker_color_set_std = np.std(marker_color_set)
    marker_color_std_tol = 15
    if marker_color_set_std >= marker_color_std_tol:
        raise Exception("CHANGE BORDER COLOR MODULE: failed to get consistent border color based on four markers.")
    
    border_mask = np.zeros(img_gray.shape, np.uint8)
    tol_min = 10
    for marker_coords in marker_set:
        curr_mask = flood(img_gray, marker_coords, tolerance = 5)
        border_mask = np.maximum(curr_mask, border_mask)
    fg_mask = 1 - border_mask
    fg_label = label(fg_mask)
    fg_prop = regionprops(fg_label)
    fg_prop.sort(key=lambda x:x.area,reverse=True)
    fg_mask = (fg_label == fg_prop[0].label)
    
    border_smoothing_size = 5
    disk_ele = disk(radius = border_smoothing_size)
    fg_mask = binary_closing(fg_mask, disk_ele)
    tmp_mask = 1 - fg_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,  50))
    
    center, major_axis_length=_get_center_by_edge(tmp_mask)
    radius=_get_radius_by_mask_center(tmp_mask,center)
    h,w = img_shape
    s_h = max(0,int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h-s_h,2 * radius), min(w-s_w,2 * radius))
    circle_mask=_get_circle_by_center_bbox(img_shape,(h//2,w//2),bbox,radius)
    
    if np.sum(fg_mask) > np.sum(circle_mask):
        result_mask = fg_mask
    else:
        result_mask = circle_mask
    #border_mask = (result_mask==0).astype(np.uint8)

    output_rgb = deepcopy(img_rgb)
    for img_channel_idx in range(3):
        img_channel = img_rgb[:, :, img_channel_idx]
        img_channel[result_mask <=0.5] = 255
        output_rgb[:, :, img_channel_idx] = img_channel
    return output_rgb, (result_mask*255).astype(np.uint8)

if __name__ == '__main__':
    image_dir = 'random_1000_model_v2'
    all_images = os.listdir(image_dir)
    
    out_dir = 'out_1000_model_v9'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    for name in all_images:
        print('Processing {}.'.format(name))
        
        try:
            img = cv2.imread(os.path.join(image_dir,name))
            output, mask = change_border_color_to_white(img)
            
            cv2.imwrite(os.path.join(out_dir, name),output)
        except:
            continue