import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import cnn_network
import cv2 as cv




def train(dataloader, model, loss_fn, optimizer):
    
    num_samples = len(dataloader.dataset)
    model.train()
    epoch_loss = 0.0

    for batch, (image, steering, throttle) in enumerate(dataloader):
        # Combine steering and throttle into one tensor (2 columns, X rows)
        target = torch.stack((steering, throttle), -1) 
        X, y = image.to(DEVICE), target.to(DEVICE)

        # Compute prediction error
        pred = model(X)  # forward propagation
        batch_loss = loss_fn(pred, y)  # compute loss
        optimizer.zero_grad()  # zero previous gradient
        batch_loss.backward()  # back propagatin
        optimizer.step()  # update parameters
        
        batch_loss, sample_count = batch_loss.item(), (batch + 1) * len(X)
        epoch_loss = (epoch_loss*batch + batch_loss) / (batch + 1)
        print(f"loss: {batch_loss:>7f} [{sample_count:>5d}/{size:>5d}]")
        
    return epoch_loss

        

def test(dataloader, model, loss_fn):
    
    # Define a test function to evaluate model performance

    #size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for image, steering, throttle in dataloader:
            #Combine steering and throttle into one tensor (2 columns, X rows)
            target = torch.stack((steering, throttle), -1) 
            X, y = image.to(DEVICE), target.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: {test_loss:>8f} \n")

    return test_loss



class BearCartDataset(Dataset): 
    """
    Customize dataset class
    """

    def __init__(self, annotation_path, image_dir, transform=transforms.ToTensor()):
        self.img_notes = pd.read_csv(annotation_path, header=None)
        self.img_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_notes)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_notes.iloc[idx, 0])
        img_arr = cv.imread(img_path, cv.IMREAD_COLOR)
        steering = self.img_notes.iloc[idx, 1].astype(np.float32)
        throttle = self.img_notes.iloc[idx, 2].astype(np.float32)
        image_tensor = transforms.ToTensor(img_arr).float
        return image_tensor, steering, throttle


# SETUP
# Designate processing unit for CNN training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")
# Create a dataloaders
data_dir = os.path.join(sys.path[0], "data", "2023_10_12_14_52")
annotation_path = os.path.join(data_dir, "annotations.csv")
image_dir = os.path.join(data_dir, "images/")
dataset = BearCartDataset(annotation_path, image_dir)
print("data length: ", len(dataset))

# # Define the size for train and test data
# train_data_len = len(collected_data)
# train_data_size = round(train_data_len*0.9)
# test_data_size = train_data_len - train_data_size 
# print("len and train and test: ", train_data_len, " ", train_data_size, " ", test_data_size)
#
# # Load the datset (split into train and test)
# train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
# train_dataloader = DataLoader(train_data, batch_size=125)
# test_dataloader = DataLoader(test_data, batch_size=125)
#
#
# # Initialize the model
# # Models that train well:
# #     lr = 0.001, epochs = 10
# #     lr = 0.0001, epochs = 15 (epochs = 20 might also work)
# model = cnn_network.DonkeyNet().to(DEVICE) # choose the architecture class from cnn_network.py
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
# epochs = 15
#
# # Optimize the model
# train_loss = []
# test_loss = []
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     training_loss = train(train_dataloader, model, loss_fn, optimizer)
#     testing_loss = test(test_dataloader, model, loss_fn)
#     print("average training loss: ", training_loss)
#     print("average testing loss: ", testing_loss)
#     # save values
#     train_loss.append(training_loss)
#     test_loss.append(testing_loss)   
#
# print(f"Optimize Done!")
#
#
# #print("final test lost: ", test_loss[-1])
# len_train_loss = len(train_loss)
# len_test_loss = len(test_loss)
# print("Train loss length: ", len_train_loss)
# print("Test loss length: ", len_test_loss)
#
#
# # create array for x values for plotting train
# epochs_array = list(range(epochs))
#
# # Graph the test and train data
# fig = plt.figure()
# axs = fig.add_subplot(1,1,1)
# plt.plot(epochs_array, train_loss, color='b', label="Training Loss")
# plt.plot(epochs_array, test_loss, '--', color='orange', label='Testing Loss')
# axs.set_ylabel('Loss')
# axs.set_xlabel('Training Epoch')
# axs.set_title('DonkeyNet 15 Epochs lr=1e-3')
# axs.legend()
# # fig.savefig('/home/robotics-j/Autobots/train_and_deploy/data/2023_10_10_15_44/DonkeyNet_15_epochs_lr_1e_3.png')
# fig.savefig(os.path.join(os.path.dirname(img_dir),'DonkeyNet_15_epochs_lr_1e_3.png'))
#
# # Save the model
# torch.save(model.state_dict(), os.path.join(os.path.dirname(img_dir),"DonkeyNet_15_epochs_lr_1e_3.pth"))

