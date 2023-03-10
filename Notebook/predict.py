#!/bin/env python3

import argparse
import os
from collections import namedtuple

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm

from dataset import EuroSAT, ImageFiles, random_split
from sklearn.metrics import classification_report, confusion_matrix

# Define a named tuple to hold the ground truth and predicted labels
TestResult = namedtuple('TestResult', 'truth predictions')

# Define a function to run the model on the specified data
@torch.no_grad()
def predict(model: nn.Module, dl: torch.utils.data.DataLoader, paths=None, show_progress=True):
    """
    Run the model on the specified data.
    Automatically moves the samples to the same device as the model.
    """

    # If show_progress is True, wrap the data loader with tqdm for a progress bar
    if show_progress:
        dl = tqdm(dl, "Predict", unit="batch")

    # Get the device of the model
    device = next(model.parameters()).device

    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to hold the predicted labels and ground truth labels
    preds = []
    truth = []

    # Initialize a counter variable for printing paths (if provided)
    i = 0

    # Iterate over the batches in the data loader
    for images, labels in dl:

        # Move the images to the same device as the model
        images = images.to(device, non_blocking=True)

        # Apply the model to the images and get the predicted labels
        p = model(images).argmax(1).tolist()

        # Append the predicted labels and ground truth labels to the corresponding lists
        preds += p
        truth += labels.tolist()

        # If paths are provided, print the predicted label for each image along with its path
        if paths:
            for pred in p:
                print(f"{paths[i]!r}, {pred}")
                i += 1

    # Create a TestResult object containing the ground truth and predicted labels as tensors
    return TestResult(truth=torch.as_tensor(truth), predictions=torch.as_tensor(preds))

# Define a function to generate a classification report and confusion matrix from a TestResult object
def report(result: TestResult, label_names):
    """
    Generate a classification report and confusion matrix from a TestResult object.
    """

    # Compute the classification report and print it
    cr = classification_report(result.truth, result.predictions, target_names=label_names, digits=3)
    print("Classification report")
    print(cr)

    # Compute the confusion matrix and print it
    confusion = confusion_matrix(result.truth, result.predictions)
    print("Confusion matrix")
    print(confusion)

    # If pandas is installed, format the confusion matrix as a data frame and print it
    try:
        import pandas as pd

        # Keep only the first three characters of the label names for the columns
        confusion = pd.DataFrame(confusion, index=label_names, columns=[s[:3] for s in label_names])
        print("Confusion matrix (formatted)")
        print(confusion)
    except ImportError:
        # If pandas is not installed, skip the formatting step and continue without crashing
        pass


def main(args):
    # Load saved model weights
    save = torch.load(args.model, map_location='cpu')

    # Load normalization parameters from saved model
    normalization = save['normalization']

    # Create a new ResNet50 model with the same number of classes as the saved model
    model = models.resnet101(num_classes=save['model_state']['fc.bias'].numel())

    # Load the saved weights into the new model
    model.load_state_dict(save['model_state'])

    # Move the model to the appropriate device (CPU or GPU)
    model = model.to(args.device)

    # Define the image transformations to be applied
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normalization)])

    # If files are specified, create an ImageFiles dataset with the specified transformations
    # Otherwise, create an instance of the EuroSAT dataset and split it into trainval and test sets
    if args.files:
        test = ImageFiles(args.files, transform=tr)
    else:
        dataset = EuroSAT(transform=tr)
        trainval, test = random_split(dataset, 0.99899, random_state=42)

    # Create a DataLoader to load images in batches for prediction
    test_dl = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    # Run prediction on the test set
    result = predict(model, test_dl, paths=args.files)

    # If files are not specified (i.e., running on the test set), analyze the results
    if not args.files:
        report(result, dataset.classes)

if __name__ == '__main__':
    # Define the parser for command line arguments
    parser = argparse.ArgumentParser(
        description="""Predict the label on the specified files and outputs the results in csv format.
            If no file is specified, then run on the test set of EuroSAT and produce a report.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add arguments to the parser
    parser.add_argument(
        '-m', '--model', default='weights/best.pt', type=str, help="Model to use for prediction"
    )
    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help="Number of workers for the DataLoader",
    )
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('files', nargs='*', help="Files to run prediction on")
    parser.add_argument('-f')

    # Parse the command line arguments
    args = parser.parse_args()

    # Set the device to run on
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
