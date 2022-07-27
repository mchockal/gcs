import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image

import pandas as pd

training_images_path = "/home/meena270593/dataset/"
training_steering_path = "/home/meena270593/dataset/interpolated.csv"

testing_images_path = "/home/meena270593/dataset/test/center"
testing_steering_path = "/home/meena270593/dataset/test/CH2_final_evaluation.csv"

image_extension = ".jpg"

class NvidiaDaveDataset(Dataset):
    """
    Provides access to the images and steering angle data present in the dataset
    used by NVIDIA's DAVE-2 end-to-end autonomous steering system.

    Code is based off the PyTorch tutorial here:

    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(self, steering_label_file, image_directory, transform=None, target_transform=None, test_set=False):
        steering_data = pd.read_csv(steering_label_file)
        if test_set:
            self.image_filenames = steering_data["frame_id"]
            self.steering_angles = steering_data["steering_angle"]
        else:
            self.image_filenames = steering_data["filename"]
            self.steering_angles = steering_data["angle"]
        self.image_directory = image_directory
        self.transform = transform
        self.target_transform = target_transform
        self.test_set = test_set

    def __len__(self):
        return len(self.steering_angles)

    def __getitem__(self, idx):
        image_filename = str(self.image_filenames.iloc[idx])

        if self.test_set:
            image_filename += image_extension

        image_path = os.path.join(self.image_directory, image_filename)
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
        nvidia_dataset = NvidiaDaveDataset(training_steering_path, training_images_path)
    else:
        nvidia_dataset = NvidiaDaveDataset(training_steering_path, training_images_path, transform=transform)

    dataset_length = len(nvidia_dataset)

    # Reserving 20% of dataset for validation.
    num_training_samples = int(dataset_length * 0.8)
    num_validation_samples = dataset_length - num_training_samples

    # Spliting NVIDIA dataset into training, validation.
    training_data, validation_data = torch.utils.data.random_split(
        nvidia_dataset, [num_training_samples, num_validation_samples],
    )

    # Converting datasets into data loaders to introduce mini-batching and per-epoch shuffling.
    training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # Pulling in the NVIDIA test dataset.
    nvidia_test_dataset = None

    if transform is None:
        nvidia_test_dataset = NvidiaDaveDataset(testing_steering_path, testing_images_path, test_set=True)
    else:
        nvidia_test_dataset = NvidiaDaveDataset(testing_steering_path, testing_images_path, transform=transform, test_set=True)

    # Creating a test data loader.
    testing_dataloader = DataLoader(nvidia_test_dataset)

    return training_dataloader, validation_dataloader, testing_dataloader
