#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import time
import timm
import torch
import sklearn.metrics

from PIL import Image

import numpy as np
import pandas as pd
import torch.nn as nn

from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, Dataset

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[2]:


metadata = pd.read_csv("SnakeCLEF2021_train_metadata_PROD.csv")
min_train_metadata = pd.read_csv("SnakeCLEF2021_min-train_metadata_PROD.csv")

print(len(metadata), len(min_train_metadata))


# In[3]:


metadata.head(5)


# In[4]:


train_metadata = min_train_metadata
val_metadata = metadata[metadata['subset'] == 'val']

print(len(train_metadata), len(val_metadata))
len(min_train_metadata.binomial.unique())


# In[5]:


N_CLASSES = 772

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['image_path'].values[idx]
        label = self.df['class_id'].values[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


# In[6]:


from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

model._fc = nn.Linear(model._fc.in_features, N_CLASSES)


# In[7]:


HEIGHT = 224
WIDTH = 224

from albumentations import Compose, Normalize, Resize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
from albumentations import RandomCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, CenterCrop, PadIfNeeded, RandomResizedCrop

def get_transforms(*, data):
    assert data in ('train', 'valid')

    if data == 'train':
        return Compose([
            RandomResizedCrop(WIDTH, HEIGHT, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(WIDTH, HEIGHT),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# In[8]:


train_dataset = TrainDataset(train_metadata, transform=get_transforms(data='train'))
valid_dataset = TrainDataset(val_metadata, transform=get_transforms(data='valid'))


# In[9]:


BATCH_SIZE = 64
EPOCHS = 50
WORKERS = 8

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)


# In[10]:


from sklearn.metrics import f1_score, accuracy_score
import tqdm


n_epochs = EPOCHS
lr = 0.01

optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5)
criterion = nn.CrossEntropyLoss()

model.to(device)

for epoch in range(n_epochs):
    start_time = time.time()

    model.train()
    avg_loss = 0.

    optimizer.zero_grad()

    for i, (images, labels) in tqdm.tqdm(enumerate(train_loader)):

        images = images.to(device)
        labels = labels.to(device)

        y_preds = model(images)
        loss = criterion(y_preds, labels)
        avg_loss += loss.item() / len(train_loader)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    avg_val_loss = 0.
    preds = np.zeros((len(valid_dataset)))

    for i, (images, labels) in enumerate(valid_loader):

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            y_preds = model(images)

        preds[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = y_preds.argmax(1).to('cpu').numpy()
        loss = criterion(y_preds, labels)
        avg_val_loss += loss.item() / len(valid_loader)
        
    scheduler.step()

    score = f1_score(val_metadata['class_id'], preds, average='macro')
    accuracy = accuracy_score(val_metadata['class_id'], preds)

    elapsed = time.time() - start_time
    print(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} F1: {score:.6f}  Accuracy: {accuracy:.6f} time: {elapsed:.0f}s')


# In[11]:


torch.save(model.state_dict(), f'SnakeCLEF2021-EfficientNet-B0_224-50E.pth')


# In[ ]:




