import cv2 
import pandas as pd
import numpy as np

import os 

def load_image(img_path):
	print(img_path)
	img = cv2.imread(img_path)


# In[9]:
df_train = pd.read_csv('trainonly_metadata.csv')

img_size = 299
batch_size = 16 #16 images per batch
#train_img_ids = df_train.image_name.values
train_img_paths = df_train.image_path.values
train_img_ids = df_train.UUID.values
n_batches = len(train_img_paths)//batch_size + 1

features = {}
for b in range(n_batches):
	start = b*batch_size
	end = (b+1)*batch_size
	batch_paths = train_img_paths[start:end]
	batch_ids = train_img_ids[start:end]
	print(batch_paths)
	print(batch_ids)
	batch_images = np.zeros((len(batch_paths),img_size,img_size,3))
	for i,img_path in enumerate(batch_paths):
		batch_images[i] = load_image(img_path)
