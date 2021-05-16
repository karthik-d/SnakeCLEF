import numpy as np
import pandas as pd 

import os

df_train = pd.read_csv('train_metadata.csv')
df_test = pd.read_csv('test_metadata.csv')

BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
TEST_PATH = os.path.abspath(os.path.join(BASE_PATH, 'Datasets', 'testset'))

print(df_train.info())
print(df_test.info())

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
from lightgbm import LGBMClassifier
import tensorflow as tf
tf.test.gpu_device_name()

device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print(device_name)
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))


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


def load_image(prefix, img_path):
	img_path = os.path.join(prefix, img_path)
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	new_img = resize_image(img)
	new_img = preprocess_input(new_img)
	return new_img


# In[9]:


img_size = 299
batch_size = 8 #8 images per batch
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
        x = Dense(256, activation='relu') (x)
        x = Dropout(0.5) (x)
        output_layer = Dense(n_out, activation='sigmoid', name='final_output') (x)
        model = Model(input=base_model.input, outputs=output_layer)
        image_size = input_shape[:2]
    elif model_name == "inceptionv3":
        print("Extract Features using "+model_name)
        base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
        model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
        image_size = (299, 299)
    elif model_name == "inceptionresnetv2":
        print("Extract Features using "+model_name)
        base_model = InceptionResNetV2(include_top=include_top,weights='imagenet',input_tensor=None,input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
        # Feature extraction from intermediate layer
        inter_x = GlobalAveragePooling2D() (base_model.output)
        inter_model = Model(inputs=base_model.input, outputs=inter_x)
        # Build rest of the model
        x = Flatten()(base_model.output) 
        y = Dense(1024, activation='relu')(x) 
        y = Dropout(0.2)(y) # 0.2
        output_layer = Dense(n_out, activation='sigmoid', name='final_output')(y)
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
    return model, inter_model


# In[11]:


# m = create_model("inceptionresnetv2", False, 'imagenet',input_shape=(299,299,3), n_out=1)
m, inter = create_model("inceptionresnetv2", False, 'imagenet',input_shape=(299,299,3), n_out=772)
# 'inter' extracts 1536 features
#m.summary()
inter.summary()

# In[12]:


features = {}
features_inter = dict()
for b in range(n_batches):
	start = b*batch_size
	end = (b+1)*batch_size
	batch_paths = train_img_paths[start:end]
	batch_ids = train_img_ids[start:end]
	batch_images = np.zeros((len(batch_paths),img_size,img_size,3))
	for i,img_path in enumerate(batch_paths):
		batch_images[i] = load_image(BASE_PATH, img_path)
	batch_preds = m.predict(batch_images)
	batch_preds_inter = inter.predict(batch_images)
	for i,img_id in enumerate(batch_ids):
		features[img_id] = batch_preds[i]
		features_inter[img_id] = batch_preds_inter[i]
	if(b%200==0):
		print("Batch", (b+1), "done")


# In[13]:
# n = 772

train_feats = pd.DataFrame.from_dict(features, orient='index', columns=['feature_{num}'.format(num=i) for i in range(772)])
train_feats.to_csv(os.path.join('new', 'train_img_features.csv'))
train_feats.head()

train_feats_inter = pd.DataFrame.from_dict(features_inter, orient='index', columns=['feature_{num}'.format(num=i) for i in range(1536)])
train_feats_inter.to_csv(os.path.join('new', 'train_img_features_inter.csv'))
train_feats_inter.head()



# In[14]:


#test_img_ids = df_test.image_name.values
#n_batches = len(test_img_ids)//batch_size + 1
test_img_paths = df_test.file_path.values
test_img_ids = df_test.UUID.values
n_batches = len(test_img_paths)//batch_size + 1

# In[15]:


features = {}
features_inter = dict()
for b in range(n_batches):
	start = b*batch_size
	end = (b+1)*batch_size
	batch_paths = test_img_paths[start:end]
	batch_ids = test_img_ids[start:end]
	batch_images = np.zeros((len(batch_paths),img_size,img_size,3))
	for i,img_path in enumerate(batch_paths):
		batch_images[i] = load_image(TEST_PATH, img_path)
	batch_preds = m.predict(batch_images)
	batch_preds_inter = inter.predict(batch_images)
	for i,img_id in enumerate(batch_ids):
		features[img_id] = batch_preds[i]
		features_inter[img_id] = batch_preds_inter[i]
	if(b%200==0):
		print("Batch", (b+1), "done")

# In[16]:

"""
test_feats = pd.DataFrame.from_dict(features, orient='index', columns=['feature_{num}'.format(num=i) for i in range(772)])
test_feats.to_csv(os.path.join('new', 'test_img_features.csv'))
test_feats.head()
"""
test_feats_inter = pd.DataFrame.from_dict(features_inter, orient='index', columns=['feature_{num}'.format(num=i) for i in range(1536)])
test_feats_inter.to_csv(os.path.join('new', 'test_img_features_inter.csv'))
test_feats_inter.head()


