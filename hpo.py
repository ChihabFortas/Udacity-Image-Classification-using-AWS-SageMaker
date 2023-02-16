# Importing required dependencies.
import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms, models

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import dependencies for Debugging andd Profiling
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    '''
          this function take a model and a testing data loader and will get
          the test accuray/loss of the model Remember to include any 
          debugging/profiling hooks that you might need
    '''

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.to(device) # need to put data on GPU device
            target=target.to(device)
            output = model(data)
            test_loss += criterion(output, target)  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, validloader, loss_criterion, optimizer, epoch, device):
    '''
        this function take a model and data loaders for training
        and will get train the model Remember to include any 
        debugging/profiling hooks that you might need
    '''
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data=data.to(device) # need to put data on GPU device
        target=target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    return model 
    
def net():
    '''
    this function that initializes the model
    Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained = True) # Use ResNet 18
    #freeze model params
    for param in model.parameters():
        param = param.requires_grad_(False)

    # New Fully Connected layers
    model.fc = nn.Sequential(
                          nn.Linear(model.fc.in_features, 256),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(256, 133),                   
                          nn.LogSoftmax(dim=1))
    return model

def create_data_loaders(data, batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # transformers
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_valid_transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])

    # datasets
    trainset = datasets.ImageFolder(os.path.join(data, "train/"), transform = train_transform)
    validset = datasets.ImageFolder(os.path.join(data, "valid/"), transform = test_valid_transform)
    testset = datasets.ImageFolder(os.path.join(data, "test/"), transform = test_valid_transform)
    
    # loaders
    trainloader = torch.utils.data.DataLoader(trainset , batch_size=batch_size , shuffle = True)
    validloader = torch.utils.data.DataLoader(validset , batch_size=test_batch_size , shuffle = True)
    testloader = torch.utils.data.DataLoader(testset  , batch_size=test_batch_size) 

    return trainloader, validloader, testloader

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    
def main(args):
    '''
    Initialize a model by calling the net function
    and move it to GPU if available
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # need GPU to run
    model=net()
    model = model.to(device)
    
    '''
    Create the loss funcion and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters() , lr = args.lr)
    
    '''
    Initialize the Hook to track the loss and Create Dataloaders
    then calling the train function to start training the model
    '''        
    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size, args.test_batch_size)
    
    for epoch in range(1, args.epochs + 1):
        model =train(model, train_loader, valid_loader, loss_criterion, optimizer, epoch, device)

        test(model, test_loader, loss_criterion, device)
    
    '''
    Save the trained model
    '''
    save_model(model, args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    Specify training args
    '''
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="b",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="e",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="l",
        help="learning rate (default: 0.01)"
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="tb",
        help="input batch size for testing (default: 100)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    args=parser.parse_args()
    
    main(args)
