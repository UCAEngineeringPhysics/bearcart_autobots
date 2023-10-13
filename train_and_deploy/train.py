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

    def __init__(self, annotation_path, image_dir):
        self.img_notes = pd.read_csv(annotation_path, header=None)
        self.img_dir = image_dir
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_notes)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_notes.iloc[idx, 0])
        img_arr = cv.imread(img_path, cv.IMREAD_COLOR)
        img_tensor = self.transform(img_arr)
        feature_image = img_tensor.float()
        target_steer = self.img_notes.iloc[idx, 1].astype(np.float32)
        target_throttle = self.img_notes.iloc[idx, 2].astype(np.float32)
        return feature_image, target_steer, target_throttle



# SETUP
# Designate processing unit for CNN training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")
# Create a dataloaders
data_dir = os.path.join(sys.path[0], "data", "2023_10_12_14_52")
annotation_path = os.path.join(data_dir, "annotations.csv")
image_dir = os.path.join(data_dir, "images/")
dataset_all = BearCartDataset(annotation_path, image_dir)
print("dataset total number of instances: ", len(dataset_all))
all_size = len(dataset_all)
train_size = round(all_size*0.9)
test_size = all_size - train_size 
print(f"train size: {train_size}, test size: {test_size}")
# Load the datset (split into train and test)
train_data, test_data = random_split(dataset_all, [train_size, test_size])
dataloader_train = DataLoader(train_data, batch_size=128)
dataloader_test = DataLoader(test_data, batch_size=128)
# Instantiate model
autopilot = cnn_network.DonkeyNet().to(DEVICE)
# Hyper-parameter
learning_rate = 3e-4
num_epochs = 2
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(autopilot.parameters(), lr=learning_rate)
# Optimize model
losses_train, losses_test = [], []
for ep in range(num_epochs):
    # Train
    print(f"Epoch {ep+1}\n-------------------------------")
    autopilot.train()
    num_used_samples = 0
    ep_loss_train = 0.
    for b, (img_tr, st_tr, th_tr) in enumerate(dataloader_train):
        targ_tr = torch.stack((st_tr, th_tr), dim=-1)
        feature_train, target_train = img_tr.to(DEVICE), targ_tr.to(DEVICE)
        pred_train = autopilot(feature_train)
        batch_loss_train = loss_fn(pred_train, target_train)
        optimizer.zero_grad()  # zero previous gradient
        batch_loss_train.backward()  # back propagation
        optimizer.step()  # update params
        num_used_samples = (b + 1) * target_train.shape[0]
        print(f"batch loss: {batch_loss_train.item()} [{num_used_samples}/{train_size}]")
        ep_loss_train = (ep_loss_train * b + batch_loss_train.item()) / (b + 1)
    losses_train.append(ep_loss_train)
    # Eval
    autopilot.eval()

    print(f"epoch {ep + 1} loss: {ep_loss_train}\n")

print(f"Optimize Done!")
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

