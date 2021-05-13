#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


#df_train = pd.read_csv('trainonly_metadata.csv')
#df_test = pd.read_csv('valonly_metadata.csv')
df_train = pd.read_csv('microtrain_metadata.csv')
df_test = pd.read_csv('microtrain_metadata.csv')

BASE_PATH = os.path.abspath(os.path.join(os.pardir, os.getcwd()))
# In[3]:


print(df_train.info())
print(df_test.info())


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.models import Model
from tensorflow.keras import Sequential
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D , Dropout, BatchNormalization
from tensorflow.keras.applications import Xception 
import keras.backend as K
#from lightgbm import LGBMClassifier
import tensorflow as tf
tf.test.gpu_device_name()


# In[5]:

device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))

# In[7]:


#Size to resize(256,256,3)
img_size = 299
def resize_image(img):
    old_size = img.shape[:2]
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1],new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0,0,0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img


# In[8]:


def load_image(img_path):
	img_path = os.path.join(BASE_PATH, img_path[1:])
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	new_img = resize_image(img)
	new_img = preprocess_input(new_img)
	return new_img


# In[9]:


img_size = 299
batch_size = 4 #16 images per batch
#train_img_ids = df_train.image_name.values
train_img_paths = df_train.image_path.values
train_img_ids = df_train.UUID.values
n_batches = len(train_img_paths)//batch_size + 1

#Model to extract image features
inp = Input((299,299,3))


# In[10]:


from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Flatten


def create_model(model_name, include_top, weights, input_shape, n_out): 
    
    input_tensor = Input(shape=input_shape) 
    include_top = include_top
    weights = weights
    IMAGE_SIZE    = (299, 299)   
    if model_name == "vgg16":
        print("Extract Features using "+model_name)
        base_model = VGG16(weights=weights)
        model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
        image_size = (224, 224)
    elif model_name == "vgg19":
        print("Extract Features using "+model_name)
        base_model = VGG19(weights=weights)
        model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
        image_size = (224, 224)
    elif model_name == "resnet50":
        print("Extract Features using "+model_name)
        base_model = ResNet50(include_top=False,
                                        weights="imagenet",
                                        input_tensor=Input(shape=input_shape))
        x = Flatten() (base_model.output)
        x = BatchNormalization() (x)
        x = Dense(256, activation='relu') (x)
        x = Dropout(0.5) (x)
        x = BatchNormalization() (x)
        x = Dense(128, activation='relu') (x)
        x = Dropout(0.5) (x)
        x = BatchNormalization() (x)
        x = Dense(64, activation='relu') (x)
        x = Dropout(0.5) (x)
        x = BatchNormalization() (x)
        output_layer = Dense(n_out, activation='sigmoid', name='final_output') (x)
        model = Model(input=base_model.input, outputs=output_layer)
        image_size = (224, 224)
    elif model_name == "inceptionv3":
        print("Extract Features using "+model_name)
        base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
        model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
        image_size = (299, 299)
    elif model_name == "inceptionresnetv2":
        print("Extract Features using "+model_name)
        base_model = InceptionResNetV2(include_top=include_top,weights='imagenet',input_tensor=None,input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
        x = Flatten()(base_model.output) 
        x = Dense(1024, activation='relu')(x) 
        x = Dropout(0.5)(x) 
        output_layer = Dense(n_out, activation='sigmoid', name='final_output')(x)
        model = Model(inputs=base_model.input, outputs=output_layer)
        image_size = (299, 299)
    elif model_name == "mobilenet":
        print("Extract Features using "+model_name)
        base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
        model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
        image_size = (224, 224)
    elif model_name == "xception":
        print("Extract Features using "+model_name)
        base_model = Xception(include_top=False, weights=weights, input_tensor=input_tensor) 
        x = GlobalAveragePooling2D()(base_model.output) 
        x = Dropout(0.5)(x) 
        x = Dense(1024, activation='relu')(x) 
        x = Dropout(0.5)(x) 
        final_output = Dense(n_out, activation='sigmoid', name='final_output')(x) 
        model = Model(input_tensor, final_output)
        #base_model = Xception(weights=weights)
        #model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
        image_size = (299, 299)
    else:
        base_model = None
    return model


# In[11]:


# m = create_model("inceptionresnetv2", False, 'imagenet',input_shape=(299,299,3), n_out=1)
m = create_model("resnet50", False, 'imagenet',input_shape=(299,299,3), n_out=1)


# In[12]:


features = {}
for b in range(n_batches):
	start = b*batch_size
	end = (b+1)*batch_size
	batch_paths = train_img_paths[start:end]
	batch_ids = train_img_ids[start:end]
	batch_images = np.zeros((len(batch_paths),img_size,img_size,3))
	for i,img_path in enumerate(batch_paths):
		try:
			batch_images[i] = load_image(img_path)
		except:
			pass
	batch_preds = m.predict(batch_images)
	for i,img_id in enumerate(batch_ids):
		features[img_id] = batch_preds[i]


# In[13]:


train_feats = pd.DataFrame.from_dict(features, orient='index')
#Save for future reference 
train_feats.to_csv('train_img_features.csv')
train_feats.head()


# In[14]:


#test_img_ids = df_test.image_name.values
#n_batches = len(test_img_ids)//batch_size + 1
test_img_paths = df_test.image_path.values
test_img_ids = df_test.UUID.values
n_batches = len(test_img_paths)//batch_size + 1

# In[15]:


features = {}
for b in range(n_batches):
	start = b*batch_size
	end = (b+1)*batch_size
	batch_paths = test_img_paths[start:end]
	batch_ids = test_img_ids[start:end]
	batch_images = np.zeros((len(batch_paths),img_size,img_size,3))
	for i,img_path in enumerate(batch_paths):
		try:
			batch_images[i] = load_image(img_path)
		except:
			pass
	batch_preds = m.predict(batch_images)
	for i,img_id in enumerate(batch_ids):
		features[img_id] = batch_preds[i]

# In[16]:


test_feats = pd.DataFrame.from_dict(features, orient='index')
test_feats.to_csv('test_img_features.csv')
test_feats.head()


# In[17]:

# CSV columns
#['Unnamed: 0', 'binomial', 'country', 'continent', 'genus', 'family',
#       'UUID', 'source', 'subset', 'class_id', 'image_path'],

df_train_full = pd.merge(df_train, train_feats, how='inner', left_on='UUID', right_index=True)
df_test_full = pd.merge(df_test, test_feats, how='inner', left_on='UUID', right_index=True)

#train = df_train_full.drop(['image_name','patient_id','diagnosis','benign_malignant'],axis=1)
#test = df_test_full.drop(['image_name','patient_id'],axis=1)
#Drop the unwanted columns
train = df_train_full.drop(['Unnamed: 0', 'binomial', 'genus', 'family', 'UUID', 'source', 'subset', 'image_path'], axis=1)
test = df_test_full.drop(['Unnamed: 0', 'binomial', 'genus', 'family', 'UUID', 'source', 'subset', 'image_path'],axis=1)

#Label Encode categorical features
train.country.fillna('unknown',inplace=True)
test.country.fillna('unknown',inplace=True)
train.continent.fillna('unknown',inplace=True)
test.continent.fillna('unknown',inplace=True)
le_country = LabelEncoder()
le_continent = LabelEncoder()
train.country = le_country.fit_transform(train.country)
test.country = le_country.transform(test.country)
train.continent = le_continent.fit_transform(train.continent)
test.continent = le_continent.transform(test.continent)


# In[18]:


folds = StratifiedKFold(n_splits= 5, shuffle=True)
oof_preds = np.zeros(train.shape[0])    # Out of fold
sub_preds = np.zeros(test.shape[0])
feature_importance_df = pd.DataFrame()
features = [f for f in train.columns if f != 'class_id']
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train['class_id'])):
    train_X, train_y = train[features].iloc[train_idx], train['class_id'].iloc[train_idx]
    valid_X, valid_y = train[features].iloc[valid_idx], train['class_id'].iloc[valid_idx]
    clf = LGBMClassifier(
        #device='gpu',
        n_estimators=1000,
        learning_rate=0.001,
        max_depth=8,
        colsample_bytree=0.5,
        num_leaves=50,
        random_state=23
    )
    print('*****Fold: {}*****'.format(n_fold))
    clf.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)], 
            eval_metric= 'auc', verbose= 20, early_stopping_rounds= 20)

    oof_preds[valid_idx] = clf.predict_proba(valid_X, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    del clf, train_X, train_y, valid_X, valid_y
    gc.collect()


# In[19]:


submission = pd.DataFrame({
    "UUID": df_test.UUID, 
    "prediction": sub_preds
})
submission.to_csv('submission.csv', index=False)

