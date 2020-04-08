import matplotlib.pyplot as plt
        
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

def validate(valid_loader, model, criterion, device, epoch, model_version):
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

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return epoch_loss