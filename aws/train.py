from __future__ import print_function # future proof
import argparse
import sys
import os
import json
import time
from datetime import datetime 

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from dl_utils import *
from train_utils import *

# import model
from model import ColorCNN

from torchvision import datasets, transforms
from dataloaders import ColorizationImageFolder

def model_fn(model_dir):
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorCNN()

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model.to(device)    
    
def _get_train_loader(batch_size, data_dir):
    print("Get data loader.")
    
    # define transformations
    train_transforms = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((256, 256))
    ])
    
    # image folders
    train_folder = ColorizationImageFolder(root=f'{data_dir}/train', model_version=1, transform=train_transforms)
    valid_folder = ColorizationImageFolder(root=f'{data_dir}/valid', model_version=1, transform=valid_transforms)
    
    # image loaders 
    train_loader = torch.utils.data.DataLoader(train_folder, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_folder, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader    



## TODO: Complete the main code
if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_version = 1
    print_every = 1
    
    # set the seed for generating random numbers
    custom_set_seed(42)
        
    # get train loader
    train_loader, valid_loader = _get_train_loader(args.batch_size, args.data_dir) # data_dir from above..
    
    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    time_meter = AverageMeter()
    
    model = ColorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    
    print('Starting training...')

    # Train model
    for epoch in range(0, args.epochs):

        start_time = time.time()

        # training
        train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            valid_loss = validate(valid_loader, model, criterion, device, 
                                  epoch, model_version)
            valid_losses.append(valid_loss)

        end_time = time.time()
        epoch_time = end_time - start_time
        time_meter.update(epoch_time)

        if epoch % print_every == (print_every - 1):
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Epoch time: {epoch_time:.2f} (avg. {time_meter.avg:.2f})')
        
    
