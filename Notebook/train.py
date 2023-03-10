#!/bin/env python3

import argparse
import os
import shutil

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm, trange

from dataset import EuroSAT, random_split
from predict import predict
import numpy as np


# Define a class called State to store the state of the training process
class State:
    # Define a class-level variable called best_acc to store the best accuracy achieved during training
    best_acc = 0
    # Define a class-level variable called writer to store the SummaryWriter object used for logging
    writer: SummaryWriter = None
    # Define a class-level variable called normalization to store the mean and standard deviation used for data normalization
    normalization = None


# Define a function called calc_normalization that calculates the mean and standard deviation of the training data
def calc_normalization(train_dl: torch.utils.data.DataLoader):
    # Create a tensor of zeros with shape (3,) and assign it to the variable mean
    mean = torch.zeros(3)
    # Create a tensor of zeros with shape (3,) and assign it to the variable m2
    m2 = torch.zeros(3)
    # Get the length of the train_dl DataLoader and assign it to the variable n
    n = len(train_dl)
    # Iterate over the train_dl DataLoader with tqdm and assign each batch of images and labels to images and labels variables respectively
    for images, labels in tqdm(train_dl, "Compute normalization"):
        # Calculate the mean of the images tensor across the 0th, 2nd, and 3rd dimensions and divide by n, and add it to the mean variable
        mean += images.mean([0, 2, 3]) / n
        # Calculate the mean of the squared images tensor across the 0th, 2nd, and 3rd dimensions and divide by n, and add it to the m2 variable
        m2 += (images ** 2).mean([0, 2, 3]) / n
    # Calculate the variance using the formula var = m2 - mean ** 2
    var = m2 - mean ** 2
    # Return the mean and the square root of the variance to be used for normalization
    return mean, var.sqrt()


def train_epoch(train_dl, model, loss, optimizer, epoch, args):
    # Set the model to train mode
    model.train()
    # Wrap the train_dl DataLoader with tqdm to show progress and set the unit to "batch"
    train_dl = tqdm(train_dl, "Train", unit="batch")
    train_loss = 0
    train_accs = 0
    # Iterate over the train_dl DataLoader
    for i, (images, labels) in enumerate(train_dl):
        # Move the images and labels to the specified device for faster processing
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        # Make predictions using the model
        preds = model(images)
        # Calculate the loss between the predictions and the labels
        _loss = loss(preds, labels)
        # Calculate the accuracy by comparing the predicted labels to the ground truth labels
        acc = (labels == preds.argmax(1)).float().mean()

        # Reset the gradients of the optimizer
        optimizer.zero_grad()
        # Backpropagate the loss through the model
        _loss.backward()
        # Update the parameters of the model using the optimizer
        optimizer.step()
        train_loss = train_loss + (_loss.item())
        train_accs = train_accs + acc
    return train_loss/i, train_accs/i

def main(args):
    # Define the EuroSAT dataset and split it into train, validation, and test datasets
    dataset = EuroSAT(
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
    )
    trainval, test_ds = random_split(dataset, 0.9, random_state=42)
    train_ds, val_ds = random_split(trainval, 0.9, random_state=7)

    # Create a data loader for the train dataset with computed normalization
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    mean, std = calc_normalization(train_dl)
    dataset.transform.transforms.append(transforms.Normalize(mean, std))
    State.normalization = {'mean': mean, 'std': std}

    # Create a data loader for the validation dataset
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )

    # Create or load the model and replace the head for our number of classes
    model = models.resnet101(pretrained=args.pretrained)
    if args.pretrained:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(args.device)
    loss = nn.CrossEntropyLoss()

    # Define the optimizer
    params = model.fc.parameters() if args.pretrained else model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    # Define the SummaryWriter and display some examples in tensorboard
    images, labels = next(iter(train_dl))
    originals = images * std.view(3, 1, 1) + mean.view(3, 1, 1)
    train_loss = 0
    train_accs=0
    list_train_acc=[]
    list_train_loss = []
    list_train_val_acc = []
    for epoch in trange(args.epochs, desc="Epochs"):
        print(epoch)
        train_loss,train_accs=train_epoch(train_dl, model, loss, optimizer, epoch, args)
        list_train_acc.append(train_accs)
        list_train_loss.append(train_loss)
        truth, preds = predict(model, val_dl)

        torch.save(
            {'normalization': State.normalization, 'model_state': model.state_dict()},
            'weights/checkpoint.pt',
        )

        val_acc = (truth == preds).float().mean()
        list_train_val_acc.append(val_acc)
        if val_acc > State.best_acc:
            print(f"New best validation accuracy: {val_acc}")
            State.best_acc = val_acc
            shutil.copy('weights/checkpoint.pt', 'weights/best.pt')
    np.save('acc_train.npy', np.array(list_train_acc))
    np.save('loss_train.npy', np.array(list_train_loss))
    np.save('acc_val.npy', np.array(list_train_val_acc))


    

if __name__ == '__main__':

    # Define a function to parse a string as a boolean value
    def parse_bool(s: str):
        """
        Parse a string as a boolean value.
        """
        # If the string is '1', 'true', or 'yes', return True
        if s.casefold() in ['1', 'true', 'yes']:
            return True
        # If the string is '0', 'false', or 'no', return False
        if s.casefold() in ['0', 'false', 'no']:
            return False
        # If the string is anything else, raise a ValueError
        raise ValueError()

    # Define an argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add command line arguments
    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help="Number of workers for the DataLoader",
    )
    parser.add_argument(
        '--epochs',
        default=15,
        type=int,
        metavar='N',
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        default=32,
        type=int,
        metavar='N',
        help="Batch size for the DataLoader",
    )
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.0001,
        type=float,
        metavar='LR',
        help="Learning rate for the optimizer",
    )
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument(
        '--wd',
        '--weight-decay',
        default=0,
        type=float,
        metavar='WD',
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        '--pretrained',
        default=True,
        type=parse_bool,
        help="Finetune a pre-trained model or train from scratch",
    )
    parser.add_argument('-f')

    # Parse the command line arguments and store them in an object
    args = parser.parse_args()

    # Set the device to GPU if available, otherwise CPU
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a directory to store the trained model weights, if it doesn't exist already
    os.makedirs('weights', exist_ok=True)
    main(args)
