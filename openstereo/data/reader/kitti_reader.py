import os

from .base_reader import BaseReader
from .base_reader_mul import BaseReader_mul

import numpy as np
import pdb
import cv2
from PIL import Image
import matplotlib.pyplot as plt



class KittiReader(BaseReader):
    def __init__(self, root, list_file, image_reader='PIL', disp_reader='PIL', right_disp=False, use_noc=False):
        super().__init__(root, list_file, image_reader, disp_reader, right_disp)
        self.use_noc = use_noc
        # assert disp_reader == 'PIL', 'Kitti Disp only support PIL format'
        # disp_reader = 'NPY'

        

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        
        left_img_path, right_img_path, disp_img_path = full_paths
        
        if self.use_noc:
            disp_img_path = disp_img_path.replace('disp_occ', 'disp_noc')
        left_img = self.image_loader(left_img_path)
        right_img = self.image_loader(right_img_path)
        # disp_img = self.disp_loader(disp_img_path) / 256.0
        # disp_img = self.disp_loader(disp_img_path)
        
        disp_np = self.disp_loader(disp_img_path) /256.0
        
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_np,
        }
        # pdb.set_trace()
        if self.return_right_disp:
            disp_img_right_path = disp_img_path.replace('c_0', 'c_1')
            disp_img_right = self.disp_loader(disp_img_right_path) / 256.0
            sample['disp_right'] = disp_img_right
        return sample

class KittimulReader(BaseReader_mul):
    def __init__(self, root, list_file, image_reader='PIL', disp_reader='PIL', seg_reader='seg', right_disp=False, use_noc=False):
        super().__init__(root, list_file, image_reader, disp_reader,seg_reader, right_disp)
        self.use_noc = use_noc
 
    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        try:
            left_img_path, right_img_path, disp_img_path, seg_img_path = full_paths  # Added seg_img_path for segmentation maps
            
        except:
            for path in full_paths:
                pdb.set_trace()
                print(path)
                print(full_paths)
                
        if self.use_noc:
            disp_img_path = disp_img_path.replace('disp_occ', 'disp_noc')

        # Load images using the appropriate loader
        # pdb.set_trace()
        left_img = self.image_loader(left_img_path)
        right_img = self.image_loader(right_img_path)
        disp_img = self.disp_loader(disp_img_path) / 256.0  # Normalize disparity
        #加载语义分割图像
        seg_img = self.seg_loader(seg_img_path)

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
            # 'seg_leftimg': seg_leftimg_resize,  # Add segmentation map to the output
            'labels': seg_img,
        }
        # print("KittimulReader sample:  ",sample.keys())
        
        # Optionally, load and add the right disparity image if needed
        if self.return_right_disp:
            disp_img_right_path = disp_img_path.replace('c_0', 'c_1')
            disp_img_right = self.disp_loader(disp_img_right_path) / 256.0  # Normalize disparity
            sample['disp_right'] = disp_img_right
        
        return sample
   
    
        



class KittiTestReader(KittiReader):
    def __init__(self, root, list_file, image_reader='PIL', disp_reader='PIL', right_disp=False, use_noc=False):
        super().__init__(root, list_file, image_reader, disp_reader, right_disp, use_noc)

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path = full_paths[:2]
        left_img = self.image_loader(left_img_path)
        right_img = self.image_loader(right_img_path)
        sample = {
            'left': left_img,
            'right': right_img,
            'name': left_img_path.split('/')[-1],
        }
        return sample


if __name__ == '__main__':
    dataset = KittiReader(root='../../data/kitti12', list_file='../../../datasets/KITTI12/kitti12_train165.txt')
    print(dataset)
    sample = dataset[0]
    print(sample['left'].shape, sample['right'].shape, sample['disp'].shape)

