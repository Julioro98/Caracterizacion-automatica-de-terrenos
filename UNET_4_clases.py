import tensorflow as tf
import numpy as np
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
!pip install tensorflow-io
import tensorflow_io as tfio
print(tf.__version__)



Cropland_train = glob.glob("/content/drive/MyDrive/Y_TRAIN_FILTRADA/*tif")
Satelitales_train = glob.glob("/content/drive/MyDrive/X_TRAIN_REDIM/*tif")
len(Satelitales_train)


Cropland_entrenamiento = []
for i in range(0, len(Cropland_train)):
 img4 = np.asarray(Image.open(Cropland_train[i]), dtype=np.uint8)
 img4 = img4.reshape(256,256,1)
 Cropland_entrenamiento.append(img4)


#Satelitales para entrenamiento en array
Satelitales_entrenamiento = []
for j in range(0, len(Satelitales_train)):
 img3=np.asarray(Image.open(Satelitales_train[j]), dtype=np.uint8)/255
 Satelitales_entrenamiento.append(img3)

dataset = tf.data.Dataset.from_tensor_slices((Satelitales_entrenamiento, Cropland_entrenamiento))

##Imagenes para TEST
Y_TEST = glob.glob("/content/drive/MyDrive/Y_test_unet/*tif")
X_TEST = glob.glob("/content/drive/MyDrive/X_test_unet/*tif")
len(Y_TEST)


y_test_dataset = []
y_train_validation = []
for k in range(0, len(Y_TEST)):
 img2 = np.asarray(Image.open(Y_TEST[k]), dtype=np.uint8)
 img2 = img2.reshape(1,256,256,1)
 y_test_dataset.append(img2)
 y_train_validation.append(img2)

X_test_dataset = []
X_train_validation = []
for h in range(0, len(X_TEST)):
 img=np.asarray(Image.open(X_TEST[h]), dtype=np.uint8)/255
 img = img[tf.newaxis, ...]                 
 X_test_dataset.append(img)
 X_train_validation.append(img)

x_dataset_val = tf.convert_to_tensor(X_train_validation, dtype=tf.float64)
y_dataset_val = tf.convert_to_tensor(y_train_validation, dtype=tf.float64)


# FUNCION: conv_block
def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size   
                  activation = 'relu',
                  padding = 'same',
                  kernel_initializer = 'he_normal')(inputs)
    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size
                  activation = 'relu',
                  padding = 'same',
                  kernel_initializer = 'he_normal')(conv)
    
    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)         
        
    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2, 2))(conv)
        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection

# FUNCION: upsampling_block
def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """
    
    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 3,    # Kernel size
                 strides = (2, 2),
                 padding = 'same')(expansive_input)
    
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,   # Number of filters
                  3,     # Kernel size
                  activation = 'relu',
                  padding = 'same',
                  kernel_initializer = 'he_normal')(merge)
    
    conv = Conv2D(n_filters,  # Number of filters
                  3,   # Kernel size
                  activation = 'relu',
                  padding = 'same',
                  kernel_initializer = 'he_normal')(conv)
    
    return conv


#  FUNCION: unet_model
def unet_model(input_size=(256, 256, 4), n_filters=32, n_classes=4):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs = Input(input_size)
    # Contracting Path (encoding)
    # Add a conv_block with the inputs of the unet_ model and n_filters
    cblock1 = conv_block(inputs, n_filters)
    # Chain the first element of the output of each block to be the input of the next conv_block. 
    # Double the number of filters at each new step
    cblock2 = conv_block(cblock1[0], n_filters * 2)
    cblock3 = conv_block(cblock2[0], n_filters * 4)
    cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob = 0.3) # Include a dropout of 0.3 for this layer
    # Include a dropout of 0.3 for this layer, and avoid the max_pooling layer
    cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob = 0.3, max_pooling = None) 
    
    # Expanding Path (decoding)
    # Add the first upsampling_block.
    # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 8)
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer. 
    # At each step, use half the number of filters of the previous block 
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)

    conv9 = Conv2D(n_filters,
                 3,
                 activation = 'relu',
                 padding = 'same',
                 kernel_initializer = 'he_normal')(ublock9)

    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
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



#learning_rate = 0.001
unet.compile(tf.keras.optimizers.Adam(
    learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['Accuracy',
                      tf.keras.metrics.OneHotIoU(num_classes= 4, target_class_ids= [0,3]),
                       tf.keras.metrics.OneHotMeanIoU(num_classes=4)
                       ])
              #metrics=['Accuracy', tf.keras.metrics.OneHotIoU(num_classes=4, target_class_ids=[0, 3])])


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


# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = unet.evaluate(dataset_test, batch_size=32)
print(results)


# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
#print("Generate predictions for 3 samples")
dict(zip(unet.metrics_names, results))

b=show_predictions(dataset_test, 12)