import torch 
import numpy as np

from torchvision import datasets
from skimage.color import rgb2lab, lab2rgb


class ColorizationImageFolder(datasets.ImageFolder):
    '''Custom images folder, which converts images to grayscale before loading'''
    
    def __init__(self, lab_version, **kw):
        self.lab_version=lab_version
        super(ColorizationImageFolder, self).__init__(**kw)
                
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)
            
            # convert to lab
            img_lab = rgb2lab(img_original / 255.0)
            
            if self.lab_version == 1:
                # output is in range [1,1] -> tanh activation
                img_lab = (img_lab + [0, 0, 0]) / [100, 128, 128] 
            elif self.lab_version == 2:
                # output is in range [0,1]
                img_lab = (img_lab + [0, 128, 128]) / [100, 255, 255]
            else:
                raise ValueError('Incorrect Lab version!!!')
                
            img_ab = img_lab[:,:,1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            
            img_gray = img_lab[:,:,0]
            img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()
                    
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img_gray, img_ab, target