"""Retinal-structure retention and inpainting transformations."""
import cv2
import pickle
import numpy as np
from skimage import morphology
from skimage.restoration import inpaint


occlusion_type = {0: 'orig_img', 1: 'retain_vessel', 2: 'retain_artery', 3: 'retain_vein', 4: 'retain_disc',
                 5: 'retain_macular_1dd', 6: 'retain_macular_2dd', 7: 'retain_all_with_macular_1dd',
                 8: 'retain_all_with_macular_2dd', 9: 'remove_vessel', 10: 'remove_artery', 11: 'remove_vein',
                 12: 'remove_disc', 13: 'remove_macular_1dd', 14: 'remove_macular_2dd',
                 15: 'remove_all_with_macular_1dd', 16: 'remove_all_with_macular_2dd',
                 17: 'remove_vessel_inpaint_biharmonic',
                 18: 'remove_artery_inpaint_biharmonic',
                 19: 'remove_vein_inpaint_biharmonic',
                 20: 'remove_disc_inpaint_biharmonic',
                 21: 'remove_macular_1dd_inpaint_biharmonic',
                 22: 'remove_macular_2dd_inpaint_biharmonic',
                 23: 'remove_all_with_macular_1dd_inpaint_biharmonic',
                 24: 'remove_all_with_macular_2dd_inpaint_biharmonic'}

def occlusion_get_processed(input_img, artery_map, vein_map, disc_map, macula_map_dd1, macula_map_dd2, curr_type,
                            inpainting_width=3):
    if artery_map is None:
        artery_map = np.zeros_like(input_img)[:, :, 0]
    if vein_map is None:
        vein_map = np.zeros_like(input_img)[:, :, 0]
    if disc_map is None:
        disc_map = np.zeros_like(input_img)[:, :, 0]
    if macula_map_dd1 is None:
        macula_map_dd1 = np.zeros_like(input_img)[:, :, 0]
    if macula_map_dd2 is None:
        macula_map_dd2 = np.zeros_like(input_img)[:, :, 0]

    if curr_type == 'orig_img':
        processed = input_img
    elif curr_type == 'retain_vessel':
        processed = input_img * (artery_map | vein_map)[:, :, np.newaxis]
    elif curr_type == 'retain_artery':
        processed = input_img * artery_map[:, :, np.newaxis]
    elif curr_type == 'retain_vein':
        processed = input_img * vein_map[:, :, np.newaxis]
    elif curr_type == 'retain_disc':
        processed = input_img * disc_map[:, :, np.newaxis]
    elif curr_type == 'retain_macular_1dd':
        processed = input_img * macula_map_dd1[:, :, np.newaxis]
    elif curr_type == 'retain_macular_2dd':
        processed = input_img * macula_map_dd2[:, :, np.newaxis]
    elif curr_type == 'retain_all_with_macular_1dd':
        processed = input_img * (artery_map | vein_map | disc_map | macula_map_dd1)[:, :, np.newaxis]
    elif curr_type == 'retain_all_with_macular_2dd':
        processed = input_img * (artery_map | vein_map | disc_map | macula_map_dd2)[:, :, np.newaxis]

    elif curr_type == 'remove_vessel':
        processed = cv2.inpaint(input_img, (artery_map | vein_map).astype(np.uint8), inpainting_width,
                                cv2.INPAINT_TELEA)
    elif curr_type == 'remove_artery':
        processed = cv2.inpaint(input_img, artery_map.astype(np.uint8), inpainting_width, cv2.INPAINT_TELEA)
    elif curr_type == 'remove_vein':
        processed = cv2.inpaint(input_img, vein_map.astype(np.uint8), inpainting_width, cv2.INPAINT_TELEA)
    elif curr_type == 'remove_disc':
        processed = cv2.inpaint(input_img, disc_map.astype(np.uint8), inpainting_width, cv2.INPAINT_TELEA)
    elif curr_type == 'remove_macular_1dd':
        processed = cv2.inpaint(input_img, macula_map_dd1.astype(np.uint8), inpainting_width, cv2.INPAINT_TELEA)
    elif curr_type == 'remove_macular_2dd':
        processed = cv2.inpaint(input_img, macula_map_dd2.astype(np.uint8), inpainting_width, cv2.INPAINT_TELEA)
    elif curr_type == 'remove_all_with_macular_1dd':
        processed = cv2.inpaint(input_img, (artery_map | vein_map | disc_map | macula_map_dd1).astype(np.uint8),
                                inpainting_width, cv2.INPAINT_TELEA)
    elif curr_type == 'remove_all_with_macular_2dd':
        processed = cv2.inpaint(input_img, (artery_map | vein_map | disc_map | macula_map_dd2).astype(np.uint8),
                                inpainting_width, cv2.INPAINT_TELEA)

    elif curr_type == 'remove_vessel_inpaint_biharmonic':
        processed = inpaint.inpaint_biharmonic(input_img, (artery_map | vein_map).astype(np.uint8), channel_axis=-1)
        processed = (processed * 255).astype(np.uint8)
    elif curr_type == 'remove_artery_inpaint_biharmonic':
        processed = inpaint.inpaint_biharmonic(input_img, artery_map.astype(np.uint8), channel_axis=-1)
        processed = (processed * 255).astype(np.uint8)
    elif curr_type == 'remove_vein_inpaint_biharmonic':
        processed = inpaint.inpaint_biharmonic(input_img, vein_map.astype(np.uint8), channel_axis=-1)
        processed = (processed * 255).astype(np.uint8)
    elif curr_type == 'remove_disc_inpaint_biharmonic':
        processed = inpaint.inpaint_biharmonic(input_img, disc_map.astype(np.uint8), channel_axis=-1)
        processed = (processed * 255).astype(np.uint8)
    elif curr_type == 'remove_macular_1dd_inpaint_biharmonic':
        processed = inpaint.inpaint_biharmonic(input_img, macula_map_dd1.astype(np.uint8), channel_axis=-1)
        processed = (processed * 255).astype(np.uint8)
    elif curr_type == 'remove_macular_2dd_inpaint_biharmonic':
        processed = inpaint.inpaint_biharmonic(input_img, macula_map_dd2.astype(np.uint8), channel_axis=-1)
        processed = (processed * 255).astype(np.uint8)
    elif curr_type == 'remove_all_with_macular_1dd_inpaint_biharmonic':
        processed = inpaint.inpaint_biharmonic(input_img, (artery_map | vein_map | disc_map
                                                           | macula_map_dd1).astype(np.uint8), channel_axis=-1)
        processed = (processed * 255).astype(np.uint8)
    elif curr_type == 'remove_all_with_macular_2dd_inpaint_biharmonic':
        processed = inpaint.inpaint_biharmonic(input_img, (artery_map | vein_map | disc_map
                                                           | macula_map_dd2).astype(np.uint8), channel_axis=-1)
        processed = (processed * 255).astype(np.uint8)
    return processed

def process_seg_file(occlusion_seg_file, input_img, vessel_dilation_size=5):
    with open(occlusion_seg_file, 'rb') as occlusion_seg_file_h:
        pickle_file = pickle.load(occlusion_seg_file_h)
        artery_map = pickle_file['artery_map']
        if artery_map is not None:
            artery_map = morphology.binary_dilation(artery_map,
                                                    footprint=np.ones((vessel_dilation_size, vessel_dilation_size)))
            artery_map = cv2.resize(artery_map.astype(np.uint8), (input_img.shape[0], input_img.shape[1]),
                                    interpolation=cv2.INTER_NEAREST) > 0.5

        vein_map = pickle_file['vein_map']
        if vein_map is not None:
            vein_map = morphology.binary_dilation(vein_map, footprint=np.ones((vessel_dilation_size, vessel_dilation_size)))
            vein_map = cv2.resize(vein_map.astype(np.uint8), (input_img.shape[0], input_img.shape[1]),
                                  interpolation=cv2.INTER_NEAREST) > 0.5

        disc_map = pickle_file['disc_map']
        if disc_map is not None:
            disc_map = cv2.resize(disc_map.astype(np.uint8), (input_img.shape[0], input_img.shape[1]),
                                  interpolation=cv2.INTER_NEAREST) > 0.5

        macula_map_dd1 = pickle_file['macula_map_dd1']
        if macula_map_dd1 is not None:
            macula_map_dd1 = cv2.resize(macula_map_dd1.astype(np.uint8), (input_img.shape[0], input_img.shape[1]),
                                        interpolation=cv2.INTER_NEAREST) > 0.5

        macula_map_dd2 = pickle_file['macula_map_dd2']
        if macula_map_dd2 is not None:
            macula_map_dd2 = cv2.resize(macula_map_dd2.astype(np.uint8), (input_img.shape[0], input_img.shape[1]),
                                        interpolation=cv2.INTER_NEAREST) > 0.5
    return artery_map, vein_map, disc_map, macula_map_dd1, macula_map_dd2
