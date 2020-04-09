import matplotlib.pyplot as plt
from image_utils import combine_channels

def save_temp_results(gray_input, ab_input, lab_version, save_path=None, save_name=None):
    '''
    Show/save rgb image from grayscale and ab channels
    Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}
    '''
    
    plt.clf() # clear matplotlib 
    
    gray_output, color_output = combine_channels(gray_input, ab_input, lab_version)
    
    if save_path is not None and save_name is not None: 
        plt.imsave(arr=gray_output, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_output, fname='{}{}'.format(save_path['colorized'], save_name))
        
def train(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0
    
    for i, (input_gray, input_ab, target) in enumerate(train_loader):

        optimizer.zero_grad()
        
        input_gray = input_gray.to(device)
        input_ab = input_ab.to(device)
    
        # Forward pass
        output_ab = model(input_gray) 
        loss = criterion(output_ab, input_ab) 
        running_loss += loss.item() * input_gray.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate(valid_loader, model, criterion, save_images, gray_path, color_path, device, epoch, lab_version):
    model.eval()
    running_loss = 0
    already_saved_images = False
    
    for i, (input_gray, input_ab, target) in enumerate(valid_loader):
    
        input_gray = input_gray.to(device)
        input_ab = input_ab.to(device)

        # Forward pass and record loss
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        running_loss += loss.item() * input_gray.size(0)

        # Save images to file
        if save_images and not already_saved_images:
            already_saved_images = True
            for j in range(min(len(output_ab), 2)): # save at most 2 images
                save_path = {'grayscale': gray_path, 
                             'colorized': color_path}
                save_name = f'img-{i * valid_loader.batch_size + j}-epoch-{epoch}.jpg'
                save_temp_results(input_gray[j], 
                                  ab_input=output_ab[j].detach(), 
                                  lab_version=lab_version,
                                  save_path=save_path, 
                                  save_name=save_name)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return epoch_loss