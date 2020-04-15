# libraries
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import random
import torch
import shutil
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import resize
from image_utils import combine_channels

class AverageMeter(object):
    '''
    A class borrowed from the PyTorch ImageNet tutorial.
    Used for storing the metrics over epochs.
    ''' 
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def custom_set_seed(seed):
    '''
    A function for making the PyTorch calculations as deterministic as possible.

    Parameters
    ----------
    seed : int
        A custom seed
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def plot_losses(train_losses, valid_losses):
    '''
    Helper function for plotting losses over epochs.

    Parameters
    ----------
    train_losses : array_like
        Losses from the training set
    valid_losses : array_like 
        Losses from the validation set
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')

def save_checkpoint(state, is_best=False, filename=None, path=None):
    '''
    Function for saving the checkpoint of a PyTorch NN.
    The best model has a non-customizable name. 
    Use filename only when saving intermediate results.

    Parameters
    ----------
    state : dict
        A dictionary containing the information we want to save
    is_best : bool
        A boolean flag indicating whether the model is the one that has the lowest validation loss
    filename : string
        Name of the file where we want to save the checkpoint, only use then not the best checkpoint
    path : str
        Path where to save the model
    '''
    
    if path:
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        path = ''
        
    if filename:
        filename = os.path.join(path, filename)
        torch.save(state, filename)

    if is_best:
        best_path = os.path.join(path, 'model_best.pth.tar')
        torch.save(state, best_path)
        
class Upsample(nn.Module):
    '''
    A class used for non-learnable upsampling of the images.
    For details please see the documentation of `nn.functional.interpolate`

    Parameters
    ----------
    scale_factor : int
        The scale for upsampling, default = 2
    mode : str
        The mode of upsampling, default = 'nearest'
    '''
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
        
def show_model_results(model, model_name, lab_version, path, img_size, device):
    '''
    Function for displaying the colorization done by the model. Displays the 
    grayscale image, original RGB and the colorized one.

    Parameters
    ----------
    model : PyTorch model from nn.Module
        The trained NN
    model_name : str 
        The name we want to display above the colorized image
    lab_version : int
        The variant of the lab scaling, see `combine_channels` for more details 
    path : str
        Path to the image to display 
    img_size : int
        Image size for potential resizing, must be the same one as used for training the model
    device : str
        String containing the device, one of ['cpu', 'cuda']
    '''
    
    assert device in ['cpu', 'cuda'], 'Invalid device!'

    test_image = imread(path)
    test_image = resize(test_image, (img_size, img_size))
    test_image_gray = rgb2gray(test_image)
    
    gray_tensor = torch.from_numpy(test_image_gray).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        ab_tensor = model(gray_tensor)
        
    _, color_output = combine_channels(gray_tensor[0], ab_tensor[0], lab_version)
        
    fig, ax = plt.subplots(1, 3, figsize = (12, 15))

    imshow(test_image_gray, ax=ax[0])
    ax[0].axis('off')
    ax[0].set_title('Grayscale')
    
    imshow(color_output, ax=ax[1])
    ax[1].axis('off')
    ax[1].set_title(model_name)
    
    imshow(test_image, ax=ax[2]) 
    ax[2].axis('off')
    ax[2].set_title('Ground Truth (RGB)')

    fig.show()

def compare_colorization(model_list, model_names, lab_version, path, img_size, device, save_dir=None):
    '''
    Function for displaying the colorization done by various models. Displays n+1 images, 
    where n is len(model_list). The extra image is the original RGB image.

    Parameters
    ----------
    model_list : list
        List of trained PyTorch models
    model_names : list 
        List of model names
    lab_version : int
        The variant of the lab scaling, see `combine_channels` for more details 
    path : str
        Path to the image to display 
    img_size : int
        Image size for potential resizing, must be the same one as used for training the model
    device : str
        String containing the device, one of ['cpu', 'cuda']
    '''
    
    assert device in ['cpu', 'cuda'], 'Invalid device!'

    test_image = imread(path)
    test_image = resize(test_image, (img_size, img_size))
    test_image_gray = rgb2gray(test_image)
    
    gray_tensor = torch.from_numpy(test_image_gray).unsqueeze(0).unsqueeze(0).float().to(device)
    
    fig, ax = plt.subplots(1, 4, figsize = (12, 16))
    
    imshow(test_image, ax=ax[0])
    ax[0].axis('off')
    ax[0].set_title('RGB')
    

    with torch.no_grad():
        
        for loc, model in enumerate(model_list, 1):
            ab_tensor = model(gray_tensor)
            _, color_output = combine_channels(gray_tensor[0], ab_tensor[0], lab_version)
        
            imshow(color_output, ax=ax[loc])
            ax[loc].axis('off')
            ax[loc].set_title(model_names[loc-1])

    if save_dir:
        os.makedirs(to_dir, exist_ok=True)
    
    fig.show()
