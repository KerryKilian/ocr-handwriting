import torch

from models import *

# change these parameters for training AND later analysis

# target size of images
target_size = 128

# max number of images per character
number_image_per_directory = 5000

batch_size = 64

num_epochs = 10

learning_rate = 0.001 

momentum = 0.99

num_workers = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_epoch = 0

# save name for architecture
architecture = "ResNeXT50"

chosen_model = ResNeXT50()

# model-targetsize-images-batch_size-num_epochs-learning_rate-momentum-num_workers
model_name = f"{architecture}-{target_size}-{number_image_per_directory}-{batch_size}-{num_epochs}-001-99-{num_workers}"
