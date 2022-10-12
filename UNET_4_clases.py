!pip install tensorflow-io
import tensorflow as tf
import numpy as np
import os
import pandas as pd 
import glob
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow import keras
from tensorflow.keras import layers
import imageio
from test_utils import summary, comparator
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow_io as tfio
print(tf.__version__)


# CARGAR LISTADO DE IMAGENES CROPLAND Y SATELITALES 
Cropland_train = glob.glob("/content/drive/MyDrive/Y_TRAIN_FILTRADA/*tif")
Satelitales_train = glob.glob("/content/drive/MyDrive/X_TRAIN_REDIM/*tif")
len(Satelitales_train)

# CREACION DE ARREGLO DE IMAGENS CROPLAND PARA ENTRENAMIENTO
Cropland_entrenamiento = []
for i in range(0, len(Cropland_train)):
 img4 = np.asarray(Image.open(Cropland_train[i]), dtype=np.uint8)
 Cropland_entrenamiento.append(img4)

#CREACION DE ARREGLO DE IMAGENES SATELITALES PARA ENTRENAMIENTO 
Satelitales_entrenamiento = []
for j in range(0, len(Satelitales_train)):
 img3=np.asarray(Image.open(Satelitales_train[j]), dtype=np.uint8)
 Satelitales_entrenamiento.append(img3)

# CREACION DEL DATASET PARA ENTRENAMIENTO
dataset = tf.data.Dataset.from_tensor_slices((Satelitales_entrenamiento, Cropland_entrenamiento))

# CARGAR LIsTADO DE IMAGENES CROPLNAD Y SATELITALES PARA TEST
Y_TEST = glob.glob("/content/drive/MyDrive/Y_test_unet/*tif")
X_TEST = glob.glob("/content/drive/MyDrive/X_test_unet/*tif")
len(Y_TEST)

# CREACION DE ARREGLO DE IMAGENES CROPLAND PARA TEST
y_test_dataset = []

for k in range(0, len(Y_TEST)):
 img2 = np.asarray(Image.open(Y_TEST[k]), dtype=np.uint8)
 y_train_dataset.append(img2)

# CREACION DE ARREGLO DE IMAGENES SATELITALES PARA TEST
X_test_dataset = []

for h in range(0, len(X_TEST)):
 img=np.asarray(Image.open(X_TEST[h]), dtype=np.uint8)    
 X_train_dataset.append(img)

x_dataset_train = tf.convert_to_tensor(X_train_validation, dtype=tf.float64)
y_dataset_train = tf.convert_to_tensor(y_train_validation, dtype=tf.float64)


# FUNCION: conv_block
def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Bloque convolucional de downsampling
    
    """

    conv = Conv2D(n_filters, 
                  3,   
                  activation = 'relu',
                  padding = 'same',
                  kernel_initializer = 'he_normal')(inputs)
    conv = Conv2D(n_filters, 
                  3,   
                  activation = 'relu',
                  padding = 'same',
                  kernel_initializer = 'he_normal')(conv)
    
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)         
        
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2, 2))(conv)
        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection

# FUNCION: upsampling_block
def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Bloque convolucional de upsampling

    """
    
    up = Conv2DTranspose(
                 n_filters,  
                 3,    
                 strides = (2, 2),
                 padding = 'same')(expansive_input)
    

    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters, 
                  3,    
                  activation = 'relu',
                  padding = 'same',
                  kernel_initializer = 'he_normal')(merge)
    
    conv = Conv2D(n_filters,  
                  3,  
                  activation = 'relu',
                  padding = 'same',
                  kernel_initializer = 'he_normal')(conv)
    
    return conv


#  FUNCION: unet_model
def unet_model(input_size=(256, 256, 4), n_filters=32, n_classes=4):
    """
    Creacion del modelo unet
    
    """
    inputs = Input(input_size)
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters * 2)
    cblock3 = conv_block(cblock2[0], n_filters * 4)
    cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob = 0.3) 
    cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob = 0.3, max_pooling = None) 
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)

    conv9 = Conv2D(n_filters,
                 3,
                 activation = 'relu',
                 padding = 'same',
                 kernel_initializer = 'he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding = 'same')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

import outputs
img_height = 256
img_width = 256
num_channels = 4

unet = unet_model((img_height, img_width, num_channels))
comparator(summary(unet), outputs.unet_model_output)
unet.summary()


unet.compile(tf.keras.optimizers.Adam(
    learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['Accuracy',
                      tf.keras.metrics.OneHotIoU(num_classes= 4, target_class_ids= [0,3]),
                       tf.keras.metrics.OneHotMeanIoU(num_classes=4)
                       ])



callbacks = [
    keras.callbacks.EarlyStopping(
      monitor='Accuracy',
      mode='max',
      min_delta=1e-3,
      patience=26,
      verbose=1,
      restore_best_weights=True),    
    keras.callbacks.ReduceLROnPlateau(
      monitor='Accuracy',
      mode='max',
      factor=0.1,
      patience=7,
      min_lr=0.000000000001)
    ]


EPOCHS = 1000
VAL_SUBSPLITS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
dataset.batch(BATCH_SIZE)
train_dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(dataset.element_spec)
model_history = unet.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


plt.plot(model_history.history["Accuracy"])

plt.plot(model_history.history["one_hot_io_u"])

plt.plot(model_history.history["one_hot_mean_io_u_1"])


def show_predictions(dataset=None, num=1):
    """
    Displays the first image of each of the num batches
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
             create_mask(unet.predict(sample_image[tf.newaxis, ...]))])


a=show_predictions(train_dataset, 12)


print("Evaluate on test data")
results = unet.evaluate(dataset_test, batch_size=32)
print(results)


# Generacion de predicciones

dict(zip(unet.metrics_names, results))

b=show_predictions(dataset_test, 12)
