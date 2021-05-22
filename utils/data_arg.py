from PIL import Image
import torchvision.transforms as transforms
from os.path import splitext
from os import listdir
import numpy as np
import os

dir_img = 'D:\\US_kidney\png_mini_mini\slice_png'
dir_mask ='D:\\US_kidney\png_mini_mini\mask_png'

def read(path,flag):
    if flag==1:
        img = Image.open(path).convert('RGB')
    else:
        img = Image.open(path).convert('L')
    return img


for i in range(1,657):
    mask_file = os.path.join(dir_mask, '{}.png'.format(i))
    img_file = os.path.join(dir_img, '{}.png'.format(i))
    mask=read(mask_file,0)
    img=read(img_file,1)
    new_mask=transforms.transforms.ColorJitter(brightness=0.2)(mask)  #ColorJitter(brightness=1)亮度
    new_img=transforms.transforms.ColorJitter(brightness=0.2)(img)
    new_mask.save(os.path.join(dir_mask, '{}.png'.format(i+656)))
    new_img.save(os.path.join(dir_img, '{}.png'.format(i +656)))






