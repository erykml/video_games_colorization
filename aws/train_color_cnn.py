# libraries
from __future__ import print_function # future proof
import argparse
import sys
import os
import json
import time
from datetime import datetime 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

from dl_utils import *
from train import *
from models import *
from dataloaders import ColorizationImageFolder
        
def _get_train_loader(img_size, batch_size, lab_version, data_dir):
    print("Getting the data loaders...")
    
    # define transformations
    train_transforms = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    valid_transforms = transforms.Compose([
        transforms.CenterCrop((img_size, img_size))
    ])
    
    # image folders
    train_folder = ColorizationImageFolder(root=f'{data_dir}/train', 
                                           lab_version=lab_version, 
                                           transform=train_transforms)
    valid_folder = ColorizationImageFolder(root=f'{data_dir}/valid', 
                                           lab_version=lab_version, 
                                           transform=valid_transforms)
    
    # image loaders 
    train_loader = torch.utils.data.DataLoader(train_folder, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_folder, 
                                               batch_size=batch_size, 
                                               shuffle=False)
    
    print('Done!')

    return train_loader, valid_loader    

    
if __name__ == '__main__':
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lab-version', type=int, default=1, metavar='N',
                        help='version of lab scaling (default: 1)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='seed (default: 42)')
    parser.add_argument('--img-size', type=int, default=224, metavar='N',
                        help='terget image size (default: 224)')
    parser.add_argument('--model-version', type=str, metavar='N',
                        help='Version of the model to train, str')
    parser.add_argument('--print-every', type=int, default=1, metavar='N',
                        help='Print epoch summary every x epochs')
    parser.add_argument('--save-every', type=int, default=5, metavar='N',
                        help='Save a checkpoint after every x epochs.')
    
    args = parser.parse_args()

    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_set_seed(args.seed)
        
    # get data loaders
    train_loader, valid_loader = _get_train_loader(args.img_size, 
                                                   args.batch_size, 
                                                   args.lab_version, 
                                                   args.data_dir) 
    
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    time_meter = AverageMeter()
    
    # model setup 
    if args.model_version == 'ColorCNN_v0':
        model = ColorCNN_v0(lab_version=args.lab_version).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.model_version == ColorCNN_v1:
        model = ColorCNN_v1(lab_version=args.lab_version).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.model_version == ColorCNN_v1_small:
        model = ColorCNN_v1_small(lab_version=args.lab_version).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.model_version == ColorCNN_v2:
        model = ColorCNN_v2(lab_version=args.lab_version).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise ValueError('Incorrect model version!!!')
    
    criterion = nn.MSELoss()

    # Train model
    
    print('Starting training...')
    for epoch in range(0, args.epochs):

        start_time = time.time()

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate_short(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)
            
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'lab_version': args.lab_version,
            'model_version': args.model_version
        }
        
        if epoch % args.save_every == (args.save_every - 1):
            save_checkpoint(checkpoint, 
                            filename=f'checkpoint_epoch_{epoch}.pth.tar', 
                            path=args.model_dir)
                   
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_checkpoint(checkpoint, is_best=True, path=args.model_dir)
            print(f'Saved best checkpoint after epoch {epoch}')

        end_time = time.time()
        epoch_time = end_time - start_time
        time_meter.update(epoch_time)

        if epoch % args.print_every == (args.print_every - 1):
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Epoch time: {epoch_time:.2f} (avg. {time_meter.avg:.2f})')
            
    # save trained model, after all epochs
    save_checkpoint(checkpoint, filename='model_last_epoch.pth.tar', path=args.model_dir)
        
    
