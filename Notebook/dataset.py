import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

# URL to download EuroSAT dataset from
URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"

# MD5 hash for integrity check of downloaded dataset
MD5 = "c8fa014336c82ac7804f0398fcb19387"

# The subdirectory of the dataset containing the images
SUBDIR = '2750'

# function to randomly split a dataset into train and validation sets
def random_split(dataset, ratio=0.9, random_state=None):
    # set the random seed if specified
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    # calculate the sizes of the train and validation sets
    n = int(len(dataset) * ratio)
    # randomly split the dataset into train and validation sets
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    # reset the random seed to its previous state if specified
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split

# subclass of ImageFolder for loading the EuroSAT dataset
class EuroSAT(ImageFolder):
    def __init__(self, root='data', transform=None, target_transform=None):
        # download the EuroSAT dataset if not already present
        self.download(root)
        # set the root directory to the directory containing the images
        root = os.path.join(root, SUBDIR)
        # call the constructor of the superclass, ImageFolder
        super().__init__(root, transform=transform, target_transform=target_transform)

    # static method to download the EuroSAT dataset
    @staticmethod
    def download(root):
        # check if the zip file already exists and has the correct MD5 hash
        if not check_integrity(os.path.join(root, "EuroSAT.zip")):
            # if not, download and extract the dataset from the URL
            download_and_extract_archive(URL, root, md5=MD5)


# subclass of Dataset for loading image files given their paths
class ImageFiles(Dataset):
    """
    Generic data loader where all paths must be given
    """

    def __init__(self, paths: [str], loader=default_loader, transform=None):
        self.paths = paths
        self.loader = loader
        self.transform = transform

    # method to return the number of images in the dataset
    def __len__(self):
        return len(self.paths)
        
    # method to return the image and its label for a given index
    def __getitem__(self, idx):
        image = self.loader(self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, -1
