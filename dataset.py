import glob
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from hyperparameter import *
import numpy as np
import torch
from torch.utils.data import Dataset

def crop_image(input_image, darkness_threshold=50):
    # Get the size of the image
    height, width = input_image.shape

    # Create a list to store bounding boxes
    bounding_boxes = []

    # Loop through each pixel in the image
    for x in range(width):
        for y in range(height):
            # Get the pixel intensity
            pixel_intensity = input_image[y, x]

            # Check if the pixel intensity is below the darkness threshold
            if pixel_intensity < darkness_threshold:
                # Create a bounding box around the pixel
                bounding_box = (x, y, x + 1, y + 1)
                bounding_boxes.append(bounding_box)

    # Check if there are any bounding boxes
    if not bounding_boxes:
        raise ValueError("No pixels below the darkness threshold found.")

    # Determine the overall bounding box that encloses all pixels below the darkness threshold
    min_x = min(box[0] for box in bounding_boxes)
    min_y = min(box[1] for box in bounding_boxes)
    max_x = max(box[2] for box in bounding_boxes)
    max_y = max(box[3] for box in bounding_boxes)

    # Crop the original image based on the overall bounding box
    cropped_image = input_image[min_y:max_y, min_x:max_x]

    return cropped_image



def make_square_image(input_image, target_size = target_size):
    height, width = input_image.shape
    max_dimension = max(width, height)
    new_width = max_dimension
    new_height = max_dimension
    pad_x = max(0, (new_width - width) // 2)
    pad_y = max(0, (new_height - height) // 2)
    padded_image = np.full((new_height, new_width), 255, dtype=np.uint8)
    padded_image[pad_y:pad_y+height, pad_x:pad_x+width] = input_image
    resized_image = cv2.resize(padded_image, (target_size, target_size))
    return resized_image


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        # Use LabelEncoder to encode string labels to numerical values
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_encoded = self.labels_encoded[idx]

        # Read and process the image as in code A
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        cropped_img = crop_image(img)
        squared_img = make_square_image(cropped_img)

        # Apply the specified transformations
        if self.transform:
            squared_img = self.transform(squared_img)

        # Convert label to PyTorch tensor
        label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        return squared_img, label_tensor

def load_data(data_directory, number_image_per_directory=5000):
    all_image_paths = []
    all_labels = []

    # Reading the amount of images specified in number_image_per_directory of each train_XXX folder. 
    # For example, it takes a maximum of 5000 images for each character
        

    for character_folder in glob.glob(os.path.join(data_directory, '*/train_*')):
        label_hex = character_folder.split("_")[3]
        label_char = chr(int(label_hex, 16))
        print(character_folder)

        # Iterate up to max_images_per_folder
        for i in range(number_image_per_directory):
            # Construct the image file name based on the pattern
            image_name = f'train_{label_hex}_{i:05d}.png'
            image_path = os.path.join(character_folder, image_name)

            # Check if the file exists
            if os.path.exists(image_path):
                all_image_paths.append(image_path)
                all_labels.append(label_char)
    
    return all_image_paths, all_labels