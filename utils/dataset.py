import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image

import pandas as pd

images_path = "../dataset"
steering_path = "../dataset/interpolated.csv"

class NvidiaDaveDataset(Dataset):
    """
    Provides access to the images and steering angle data present in the dataset
    used by NVIDIA's DAVE-2 end-to-end autonomous steering system.

    Code is based off the PyTorch tutorial here:

    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(self, steering_label_file, image_directory, transform=None, target_transform=None):
        steering_data = pd.read_csv(steering_label_file)
        self.image_filenames = steering_data["filename"]
        self.steering_angles = steering_data["angle"]
        self.image_directory = image_directory
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.steering_angles)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.image_directory, str(self.image_filenames.iloc[idx]),
        )
        image = read_image(image_path)
        image = image.float()
        label = self.steering_angles.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def load_nvidia_dataset(batch_size=64, transform=None):
    """
    Loads the NVIDIA dataset from the filesystem and returns its contents
    as easier to use PyTorch data loader objects.
    """

    nvidia_dataset = None

    if transform is None:
        nvidia_dataset = NvidiaDaveDataset(steering_path, images_path)
    else:
        nvidia_dataset = NvidiaDaveDataset(steering_path, images_path, transform=transform)

    dataset_length = len(nvidia_dataset)

    # Reserving 10% of dataset for testing.
    num_training_and_validation_samples = int(dataset_length * 0.9)

    # Reserving 80% of non-testing data for training and 20% for validation.
    num_training_samples = int(num_training_and_validation_samples * 0.8)
    num_validation_samples = num_training_and_validation_samples - num_training_samples
    num_testing_samples = dataset_length - num_training_and_validation_samples

    # Spliting NVIDIA dataset into training, validation, and test sets.
    training_data, validation_data, testing_data = torch.utils.data.random_split(
        nvidia_dataset, [num_training_samples, num_validation_samples, num_testing_samples],
    )

    # Converting datasets into data loaders to introduce mini-batching and per-epoch shuffling.
    training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    testing_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    return training_dataloader, validation_dataloader, testing_dataloader
