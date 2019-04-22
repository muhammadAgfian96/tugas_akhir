import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

# Import Warnings 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import tensorflow as the backend for Keras
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.callbacks import TensorBoard

# Import required libraries for cnfusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

"""=== Penyusunan dan Pengambilan Data ==="""
PATH = os.getcwd()

#Define data path
data_path = PATH + '/image vocal/main data'
data_dir_list = os.listdir(data_path)
data_dir_list

img_rows=128
img_cols=128
num_channel=1
num_epoch=20

# Define the number of classes

num_classes = 7
labels_name={'cyst':0, 'granuloma':1, 'nodule':2, 'none': 3, 'normal':4, 'papiloma':5, 'paralysis':6}

img_data_list= []
labels_list  = []

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    label = labels_name[dataset]
    for img in tqdm(img_list):
        input_img = cv2.imread(data_path + '/'+ dataset + '/'+ img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img,(img_rows,img_cols))
        img_data_list.append(input_img_resize)
        labels_list.append(label)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255

print (img_data.shape)
plt.imshow(img_data[1111,:])
plt.show()

labels = np.array(labels_list)

# print the count of number of samples for different classes
print(np.unique(labels,return_counts=True))

if num_channel==1:
    if K.image_dim_ordering()=='th':
        img_data= np.expand_dims(img_data, axis=1) 
        print (img_data.shape)
    else:
        img_data= np.expand_dims(img_data, axis=4) 
        print (img_data.shape)
        
else:
    if K.image_dim_ordering()=='th':
        img_data=np.rollaxis(img_data,3,1)
        print (img_data.shape)

# one-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print("X_train shape = {}".format(X_train.shape))
print("X_test shape = {}".format(X_test.shape))
print("y_train shape = {}".format(y_train.shape))
print("y_test shape = {}".format(y_test.shape))
print("img data shape = {}".format(img_data[0].shape))

"""==== PENGATURAN AUGMENTASI DATA ===="""

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(width_shift_range=0.1,
                  height_shift_range=0.1,
                  zoom_range=0.2,
                  shear_range=0.1,
                  horizontal_flip=True,
                  vertical_flip=True,
                  rotation_range=30,
                  fill_mode = "nearest")
datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size=20)
print(batches)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(128,128), cmap="gray")
    axs[i].axis('off')

#Visualize some images
image = X_test[22, :].reshape((128,128))
plt.imshow(image)
plt.show()

"""===== inisialisasi model CNN ====="""
#Initialising the input shape
input_shape=img_data[0].shape
# Design the CNN Sequential model
def leNet_model():
  model = Sequential()
  model.add(Conv2D(25, (3,3), input_shape =input_shape, padding='same', strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3)))
  
  model.add(Conv2D(20,(3,3), padding='same', strides=1,  activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3)))
  
  model.add(Conv2D(10,(3,3), padding='valid', strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(37, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))
  
  model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  return model

model = leNet_model()

"""===== TRAINING ===="""
# persiapan
BS=50
# SPE = len(X_train) // BS
SPE=500
print(SPE)

# mulai training
history = model.fit(X_train, y_train, batch_size=50, nb_epoch=30, verbose=1, validation_data=(X_test, y_test), shuffle =1)

# memplot hasil training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('Loss')
plt.xlabel('epoch')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('Accuracy')
plt.xlabel('epoch')

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# nge save model yang telah lu latih
model.save('your_model_name.h5')

# mengetest hasil training
# a. langsung dari X_test

image = X_test[22, :].reshape((128,128))
plt.imshow(image)
plt.show()

test_image = X_test[0:1]
print (test_image.shape)
print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

# b. langsung dari directory
test_img = cv2.imread('C:\\Users\\MANAHATI\\Desktop\\Jupyter 2\\test_cyst2.jpg')
test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
test_img = cv2.resize(test_img,(128,128))
test_img = np.array(test_img)
test_img = test_img.astype('float32')
test_img /= 255
print(test_img.shape)

image = test_img.reshape((128,128))
plt.imshow(image)
plt.show()

if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_img= np.expand_dims(test_img, axis=0)
		test_img= np.expand_dims(test_img, axis=0)
		print (test_img.shape)
	else:
		test_img= np.expand_dims(test_img, axis=3) 
		test_img= np.expand_dims(test_img, axis=0)
		print (test_img.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_img=np.rollaxis(test_img,2,0)
		test_img= np.expand_dims(test_img, axis=0)
		print (test_img.shape)
	else:
		test_img= np.expand_dims(test_img, axis=0)
		print (test_img.shape)
		
# Predicting the test image
print((model.predict(test_img)))
print(model.predict_classes(test_img))
if (model.predict_classes(test_img)==0):
    print("cyst")
if (model.predict_classes(test_img)==1):
    print("granuloma")
if (model.predict_classes(test_img)==2):
    print("nodule")
if (model.predict_classes(test_img)==3):
    print("none")
if (model.predict_classes(test_img)==4):
    print("normal")
if (model.predict_classes(test_img)==5):
    print("papilomaa")
if (model.predict_classes(test_img)==5):
    print("paralysis")
    
 # Print the confusion matrix
Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred,axis=1)
print(y_pred)
target_names=['Class 0 (cyst)', 'Class 1 (granuloma)', 'Class 2 (nodule)', 'Class 3 (none)',
              'Class 4 (normal)', 'Class 5 (papiloma)', 'Class 6 (paralysis)']
print(classification_report(np.argmax(y_test,axis=1),y_pred,target_names=target_names))

print('Confusion Matrix \n')
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

"""menampilkan/memvizualisasikan feature maps"""

def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations
layer_num=7     # nanti ini diganti2
filter_num=0    # ini juga di ganti2
activations = get_featuremaps(model, int(layer_num),test_img)
print (np.shape(activations))

feature_maps = activations[0][0]
# print (len(feature_maps))
for i in range(len(feature_maps)):
    print(feature_maps[i],end=',')
    
fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')
num_of_featuremaps=feature_maps.shape[0]
print(num_of_featuremaps)
fig=plt.figure(figsize=(7,7))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))

for i in range(int(num_of_featuremaps)):
    ax = fig.add_subplot(subplot_num, subplot_num, i+1)
# 	ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
    ax.imshow(feature_maps[:,:,i],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.show()
