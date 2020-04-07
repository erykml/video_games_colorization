import numpy as np
import torch

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
        
def preview_dataloader_lab(data_loader):

    # obtain one batch of training images
    data_iter = iter(data_loader)
    img_gray, img_ab, target = data_iter.next()

    # preview first 3 images in the batch
    fig, ax = plt.subplots(3, 3, figsize = (12, 15))

    for i in range(0, 3):
        imshow(img_gray[i][0].numpy(), ax=ax[i][0]) 
        ax[i][0].axis('off')
        ax[i][0].set_title('L')

        ax[i][1].imshow(img_ab[i][0].numpy(), cmap='RdYlGn_r') 
        ax[i][1].axis('off')
        ax[i][1].set_title('a')

        ax[i][2].imshow(img_ab[i][1].numpy(), cmap='YlGnBu_r') 
        ax[i][2].axis('off')
        ax[i][2].set_title('b');
        
    return img_gray, img_ab

def combine_channels(gray_input, ab_input, model_version):
    
    if gray_input.is_cuda: gray_input = gray_input.cpu()
    if ab_input.is_cuda: ab_input = ab_input.cpu()
    
    # combine channels
    color_image = torch.cat((gray_input, ab_input), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    
    # reverse the transformation from DataLoaders
    if model_version == 1:
        color_image = color_image * [100, 128, 128]
    elif model_version == 2:
        color_image = color_image * [100, 255, 255] - [0, 128, 128]
    else:
        raise ValueError('Incorrect model version!!!')
    
    # prepare the grayscale/RGB imagers
    color_output = lab2rgb(color_image.astype(
        np.float64))
    gray_output = gray_input.squeeze().numpy()
    
    return gray_output, color_output