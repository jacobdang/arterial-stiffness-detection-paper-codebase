#!/usr/bin/env python3
"""Prepare retinal-structure image variants from an RGB image and approved masks."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--masks',type=Path,required=True,help='NPZ with artery_map, vein_map, disc_map, macula_map_dd1, macula_map_dd2')
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+',default=[1,2,3,4,5,6,8,17,18,19,20,21,22,24])
    a=p.parse_args()
    import numpy as np
    from PIL import Image
    from masking import occlusion_get_processed,occlusion_type
    rgb=np.asarray(Image.open(a.image).convert('RGB'))
    if rgb.shape[:2]!=(384,384):p.error('Supply the preprocessed 384 x 384 image')
    with np.load(a.masks,allow_pickle=False) as data:
        maps=[np.asarray(data[k],dtype=bool) for k in ['artery_map','vein_map','disc_map','macula_map_dd1','macula_map_dd2']]
    if any(x.shape!=rgb.shape[:2] for x in maps):p.error('Masks must match the image dimensions')
    for index in a.indices:
        if index not in occlusion_type:p.error('Invalid occlusion index')
        name=occlusion_type[index];out=a.output_root/name/(a.image.stem+'.png')
        out.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(occlusion_get_processed(rgb,*maps,name)).save(out)
    return 0
if __name__=='__main__':raise SystemExit(main())
