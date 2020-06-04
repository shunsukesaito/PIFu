import os
import cv2
import numpy as np 

from pathlib import Path
import argparse

def get_bbox(msk):
    rows = np.any(msk, axis=1)
    cols = np.any(msk, axis=0)
    rmin, rmax = np.where(rows)[0][[0,-1]]
    cmin, cmax = np.where(cols)[0][[0,-1]]

    return rmin, rmax, cmin, cmax

def process_img(img, msk, bbox=None):
    if bbox is None:
        bbox = get_bbox(msk > 100)
    cx = (bbox[3] + bbox[2])//2
    cy = (bbox[1] + bbox[0])//2

    w = img.shape[1]
    h = img.shape[0]
    height = int(1.138*(bbox[1] - bbox[0]))
    hh = height//2

    # crop
    dw = min(cx, w-cx, hh)
    if cy-hh < 0:
        img = cv2.copyMakeBorder(img,hh-cy,0,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])    
        msk = cv2.copyMakeBorder(msk,hh-cy,0,0,0,cv2.BORDER_CONSTANT,value=0) 
        cy = hh
    if cy+hh > h:
        img = cv2.copyMakeBorder(img,0,cy+hh-h,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])    
        msk = cv2.copyMakeBorder(msk,0,cy+hh-h,0,0,cv2.BORDER_CONSTANT,value=0)    
    img = img[cy-hh:(cy+hh),cx-dw:cx+dw,:]
    msk = msk[cy-hh:(cy+hh),cx-dw:cx+dw]
    dw = img.shape[0] - img.shape[1]
    if dw != 0:
        img = cv2.copyMakeBorder(img,0,0,dw//2,dw//2,cv2.BORDER_CONSTANT,value=[0,0,0])    
        msk = cv2.copyMakeBorder(msk,0,0,dw//2,dw//2,cv2.BORDER_CONSTANT,value=0)    
    img = cv2.resize(img, (512, 512))
    msk = cv2.resize(msk, (512, 512))

    kernel = np.ones((3,3),np.uint8)
    msk = cv2.erode((255*(msk > 100)).astype(np.uint8), kernel, iterations = 1)

    return img, msk

def main():
    '''
    given foreground mask, this script crops and resizes an input image and mask for processing.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image', type=str, help='if the image has alpha channel, it will be used as mask')
    parser.add_argument('-m', '--input_mask', type=str)
    parser.add_argument('-o', '--out_path', type=str, default='./sample_images')
    args = parser.parse_args()

    img = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        msk = img[:,:,3:]
        img = img[:,:,:3]
    else:
        msk = cv2.imread(args.input_mask, cv2.IMREAD_GRAYSCALE)

    img_new, msk_new = process_img(img, msk)

    img_name = Path(args.input_image).stem

    cv2.imwrite(os.path.join(args.out_path, img_name + '.png'), img_new)
    cv2.imwrite(os.path.join(args.out_path, img_name + '_mask.png'), msk_new)

if __name__ == "__main__":
    main()