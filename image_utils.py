import numpy as np

from skimage.color import rgb2lab, rgb2gray, lab2rgb
from skimage.io import imread, imshow

import matplotlib.pyplot as plt

def preview_image(path, is_path=True):
    '''
    Function for viewing a preview of an image
    '''
    if not is_path:
        image = path
    else:
        image = imread(path)
    print('The size of the image is:', image.shape)
    imshow(image)
    
def preview_lab_image(path, is_path=True):
    '''
    Function for viewing a preview of an image in the lab scale
    '''
    if not is_path:
        image = path
    else:
        image = imread(path)
    image_lab = rgb2lab(image / 255)
    image_lab = (image_lab + [0, 128, 128]) / [100, 255, 255]
    
    fig, ax = plt.subplots(1, 4, figsize = (18, 30))

    ax[0].imshow(image_lab) 
    ax[0].axis('off')
    ax[0].set_title('Lab')

    imshow(image_lab[:,:,0], ax=ax[1]) 
    ax[1].axis('off')
    ax[1].set_title('L')

    ax[2].imshow(image_lab[:,:,1], cmap='RdYlGn_r') 
    ax[2].axis('off')
    ax[2].set_title('a')

    ax[3].imshow(image_lab[:,:,2], cmap='YlGnBu_r') 
    ax[3].axis('off')
    ax[3].set_title('b')

    plt.show()
        