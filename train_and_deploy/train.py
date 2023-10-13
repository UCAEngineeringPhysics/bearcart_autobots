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
# Create dataloaders
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
dataloader_eval = DataLoader(test_data, batch_size=128)
# Instantiate model
autopilot = cnn_network.DonkeyNet().to(DEVICE)
# Hyper-parameter
learning_rate = 1e-4
num_epochs = 15
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(autopilot.parameters(), lr=learning_rate)
# Optimize model
losses_train, losses_eval = [], []
for ep in range(num_epochs):
    # Train
    print(f"Epoch {ep+1}\n-------------------------------")
    autopilot.train()
    num_used_samples = 0
    ep_loss_train = 0.
    for b, (im_tr, st_tr, th_tr) in enumerate(dataloader_train):
        targ_tr = torch.stack((st_tr, th_tr), dim=-1)
        feature_train, target_train = im_tr.to(DEVICE), targ_tr.to(DEVICE)
        pred_train = autopilot(feature_train)
        batch_loss_train = loss_fn(pred_train, target_train)
        optimizer.zero_grad()  # zero previous gradient
        batch_loss_train.backward()  # back propagation
        optimizer.step()  # update params
        num_used_samples += target_train.shape[0]
        print(f"batch loss: {batch_loss_train.item()} [{num_used_samples}/{train_size}]")
        ep_loss_train = (ep_loss_train * b + batch_loss_train.item()) / (b + 1)
    losses_train.append(ep_loss_train)
    # Eval
    autopilot.eval()
    ep_loss_eval = 0.
    with torch.no_grad():
        for b, (im_ev, st_ev, th_ev) in enumerate(dataloader_eval):
            targ_ev = torch.stack((st_ev, th_ev), dim=-1)
            feature_eval, target_eval = im_ev.to(DEVICE), targ_ev.to(DEVICE)
            pred_eval = autopilot(feature_eval)
            batch_loss_eval = loss_fn(pred_eval, target_eval)
            ep_loss_eval = (ep_loss_eval * b + batch_loss_eval.item()) / (b + 1)
        losses_eval.append(ep_loss_eval)
    print(f"epoch {ep + 1} train loss: {ep_loss_train}, eval loss: {ep_loss_eval}\n")

# Finalize training
print(f"Optimize Done!")
pilot_title = f'{autopilot._get_name()}-{num_epochs}epochs-{str(learning_rate)}lr'
plt.plot(range(num_epochs), losses_train, 'b--', label='Training')
plt.plot(range(num_epochs), losses_eval, 'orange', label='Evaluation')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title(pilot_title)
plt.savefig(os.path.join(sys.path[0], data_dir, f'{pilot_title}.png'))
# Save the model
torch.save(autopilot.state_dict(), os.path.join(sys.path[0], data_dir, f'{pilot_title}.pth'))

