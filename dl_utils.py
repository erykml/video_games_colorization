import numpy as np
import os
import random
import torch
import matplotlib.pyplot as plt
import shutil

from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import resize

from image_utils import combine_channels

class AverageMeter(object):
    '''A handy class from the PyTorch ImageNet tutorial''' 
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def plot_losses(train_losses, valid_losses):
    '''
    Helper function for plotting losses over time
    '''
    
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
    
    plt.style.use('default')

def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar', path=None):
    
    if path:
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        path = ''
        
    filename = os.path.join(path, filename)
        
    torch.save(state, filename)
    
    if is_best:
        best_path = os.path.join(path, 'model_best.pth.tar')
        shutil.copyfile(filename, best_path)
        
def show_model_results(model, model_name, lab_version, path, img_size, device):
    
    test_image = imread(path)
    test_image = resize(test_image, (img_size, img_size))
    test_image_gray = rgb2gray(test_image)
    
    gray_tensor = torch.from_numpy(test_image_gray).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        ab_tensor = model(gray_tensor)
        
    gray_output, color_output = combine_channels(gray_tensor[0], ab_tensor[0], lab_version)
        
    fig, ax = plt.subplots(1, 3, figsize = (12, 15))

    imshow(test_image, ax=ax[0]) 
    ax[0].axis('off')
    ax[0].set_title('RGB')

    imshow(test_image_gray, ax=ax[1])
    ax[1].axis('off')
    ax[1].set_title('Grayscale')

    imshow(color_output, ax=ax[2])
    ax[2].axis('off')
    ax[2].set_title(model_name)

    fig.show()