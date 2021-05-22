from os.path import splitext
from os import listdir
import numpy as np
import os
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1,size=512):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.size=size
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale,size):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        newW, newH = int(scale * size), int(scale * size)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        #print('mask_dir',self.masks_dir)
        mask_file = os.path.join(self.masks_dir,"{}.png".format(idx))#glob(self.masks_dir + idx + '.*')
        #print('mask_file',mask_file)
        img_file = os.path.join(self.imgs_dir,"{}.png".format(idx))                  #glob(self.imgs_dir + idx + '.*')

        # assert len(mask_file) == 1, \
        #     f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        # assert len(img_file) == 1, \
        #     f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file).convert('L')
        img = Image.open(img_file).convert('RGB')

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale,self.size)
        mask = self.preprocess(mask, self.scale,self.size)
        #print('img size',img.shape)
        #print('mask size', mask.shape)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
