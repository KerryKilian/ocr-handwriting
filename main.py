# source code inspireed by
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from dataset import CustomDataset, load_data
from train_utils import plot, training, test
from hyperparameter import *

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    folder_path = os.path.join(script_directory, model_name)
    os.makedirs(folder_path, exist_ok=True)

    
    # set seed for reproducability
    torch.manual_seed(0)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # load data
    data_directory = 'all_characters/'
    all_image_paths, all_labels = load_data(data_directory, number_image_per_directory)

    print("All images read")
    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.2, random_state=42
    )

    # load train and test data
    print("Creating training Dataset")
    train_dataset = CustomDataset(image_paths=train_image_paths, labels=train_labels, transform=transforms.ToTensor())
    print("Creating test Dataset")
    test_dataset = CustomDataset(image_paths=test_image_paths, labels=test_labels, transform=transforms.ToTensor())

    # Create data loaders
    loader_params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True
    }

    # original
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, **loader_params)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, **loader_params)

    # model setup
    model = chosen_model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    criterion = nn.CrossEntropyLoss()

    train_acc_history = []
    test_acc_history = []

    train_loss_history = []
    test_loss_history = []

    best_acc = 0.0
    since = time.time()

    # train and test for each epoch
    for epoch in range(start_epoch, num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        training_loss, training_acc = training(model, train_loader, optimizer,
                                            criterion, device)
        train_loss_history.append(training_loss)
        train_acc_history.append(training_acc)

        # test
        print("Testing the model")
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print("test_loss: " + str(test_loss))
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)


        # For interrupting the code
        if test_acc > best_acc:
            best_acc = test_acc

        model_save_path = os.path.join(folder_path, f"model_epoch_{epoch}_{model_name}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'train_loss_history': train_loss_history,
            'train_acc_history': train_acc_history,
            'test_loss_history': test_loss_history,
            'test_acc_history': test_acc_history
        }, model_save_path)
        print(f'Model saved at epoch {epoch} with accuracy: {best_acc:.4f}')

    time_elapsed = time.time() - since

    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # plot loss and accuracy curves
    train_acc_history = [h.cpu().numpy() for h in train_acc_history]
    test_acc_history = [h.cpu().numpy() for h in test_acc_history]
    accuracy_save_path = os.path.join(folder_path, f"accuracy_{model_name}.png")
    loss_save_path = os.path.join(folder_path, f"loss_{model_name}.png")
    model_save_path = os.path.join(folder_path, f"{model_name}.pth")
    torch.save(model.state_dict(), model_save_path)

    plot(train_acc_history, test_acc_history, 'accuracy', num_epochs, save_path=accuracy_save_path)
    plot(train_loss_history, test_loss_history, 'loss', num_epochs, save_path=loss_save_path)

    print("test_loss_history: " + str(test_loss_history))
    print(f"Model saved to: {model_save_path}")

if __name__ == '__main__':
    main()
    