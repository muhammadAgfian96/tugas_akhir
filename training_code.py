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
