from skimage import img_as_float32, img_as_int, img_as_ubyte,img_as_uint
from skimage.io import imread
import albumentations.augmentations.transforms as transforms
import albumentations.core.composition as composition
import torch
import numpy as np 
import fundus_prep as prep

def read_img_file(filename, force_data_type_conversion=None, force_3_channel=True, force_min_img_size=None):
    # note: force_3_channel=(None/'ubyte'/'uint'/'int'). E.g., force_3_channel='ubyte'
    # setting force_data_type_conversion will perform certain image intensity scaling 
    # to ensure the output satisfying data type requirements and range convention 
    # See https://scikit-image.org/docs/stable/api/skimage.html for details
    # if force_data_type_conversion causes issues, you should disable this option by setting it to None
    #
    # note: force_3_channel=(True/False). Enabling it to output a error message if input image is not 3 channeled.
    #
    # note: force_min_img_size=(None/uint). E.g., force_min_img_size== 128
    # enabling it to output a error message if either width or height of the input image is smaller than the specified size. 
    
    if force_data_type_conversion == 'ubyte':
        img = img_as_ubyte(imread(filename))
    elif force_data_type_conversion == 'float':
        img = img_as_float32(imread(filename))
    elif force_data_type_conversion == 'uint':
        img = img_as_uint(imread(filename))
    elif force_data_type_conversion == 'int':
        img = img_as_int(imread(filename))
    elif force_data_type_conversion is None:
        img = imread(filename)
    else:
        raise ValueError('Invalid force_data_type_conversion value.')
    
    if force_3_channel and len(img.shape) != 3:
        raise ValueError('Input image must contain 3 channels as enforced by force_3_channel option.')
    
    if force_min_img_size and (img.shape[0] < force_min_img_size or img.shape[1] < force_min_img_size):
        raise ValueError('Input image size is ' + str(img.shape) + ', which is smaller than force_min_img_size=' + str(force_min_img_size))
    
    return img
    
def crop_ROI(img):
    
    '''
    Function to crop the ROI (the bubble area of the fundus image).
    Inputs:
        img: the numpy array of the input image
    Returns:
        cropped_img: the cropped numpy array of the input image
        borders: the border of the initial cropping
    '''
    
    cropped_img  = prep.process_without_gb(img)
    
    return cropped_img
    
    
def basic_resize_and_normalization(img, resized_height=None, resized_width=None, interpolation_order=3, 
                                   pixel_mean=None, pixel_std=None, max_pixel_value=None):
    # note: resized_height=(None/uint), resized_width=(None/uint) E.g., resized_height=512, resized_width=512
    # setting either resized_height or resized_width to None will disable resizing
    #
    # note: interpolation_order=(uint). For RGB or grayscale input image, recommended to use 3. 
    # For labelling mask, you MUST set interpolation_order to 0 to ensure correct resizing
    #
    # note: pixel_mean=(None/python_list), pixel_std=(None/python_list), max_pixel_value=(None/float) 
    # E.g., pixel_mean=(0.5, 0.5, 0.5), pixel_std=(0.5, 0.5, 0.5), max_pixel_value=255.0
    # Pls check https://albumentations.readthedocs.io/en/latest/api/augmentations.html?highlight=Normalize#albumentations.augmentations.transforms.Normalize
    # setting any value among pixel_mean, pixel_std, max_pixel_value to None will disable normalization
    
    composed_transform = list()
    if resized_height is not None and resized_width is not None:
        composed_transform.append(
            transforms.Resize(resized_height, resized_width, interpolation=interpolation_order))
    if pixel_mean is not None and pixel_std is not None and max_pixel_value is not None:
        composed_transform.append(
            transforms.Normalize(mean=pixel_mean, std=pixel_std, max_pixel_value=max_pixel_value))        
    
    if composed_transform:
        aug_obj = composition.Compose(composed_transform, p=1)
        aug_dic = {'image': img}
        aug_img = aug_obj(**aug_dic)
        img = aug_img['image']
    
    return img



def img_numpy_to_torch(img, force_data_type_conversion='float32', force_device=None):
    # note: force_data_type_conversion=(None/'float32'/'float'/'long'). E.g., force_data_type_conversion='float'
    # setting force_data_type_conversion will ensure the output tensor satisfying pytorch requirements
    # you usually should use float or float32. 
    # However for mask images and for certain loss (like cross-entropy), long type is needed by pytorch
    #
    # note: force_device=(None/torch.device object)
    
    img = np.transpose(img, (2, 0, 1))
    
    if force_data_type_conversion == 'float' or force_data_type_conversion == 'float32':
        img = img.astype(np.float32)
    elif force_data_type_conversion == 'long':
        img = img.astype(np.long)
    elif force_data_type_conversion is None:
        pass
    else:
        raise ValueError('Invalid force_data_type_conversion value.')
        
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img)
    
    if force_device is not None:
        img = img.to(force_device)
    return img