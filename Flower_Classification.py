import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from os import listdir
import re
from PIL import Image

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    targets = []
    for image in imagesList:
            img = np.zeros((128,128,3))
            img = np.array(Image.open(path+image).convert('RGB'))
            loadedImages.append(img)
            targets.append(int(re.split("_",image)[0]))
    return np.array(loadedImages),np.array(targets)

path = r"C:\Users\ASUS\桌面\Xcelerate\CNN\flower color images\Flower_Classification\flowers\\"

imgs,targets = loadImages(path)

classes = np.unique(targets)
nClasses = len(classes)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
data_augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(height_factor=(-0.1, 0.1))
    ]
)

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
for i in range(9):
    image = imgs[0]
    plt.subplot(3, 3, i+1)
    augmented_images = data_augment(image)
    plt.imshow(augmented_images.astype('uint8'))
    plt.axis('off')

from tensorflow.keras import regularizers


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    aug = data_augment(inputs)
    normed = layers.Rescaling(1 / 255.)(aug)

    x = layers.Activation("linear")(normed)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 5, padding='valid', input_shape=input_shape)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, 3, padding='valid', kernel_regularizer=regularizers.L2(0.00005))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, 3, padding='valid', kernel_regularizer=regularizers.L2(0.00005))(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

model = make_model(input_shape=(128, 128, 3), num_classes=20)
# keras.utils.plot_model(model, show_shapes=True)
print(model.summary())

from sklearn.model_selection import train_test_split
X_train, X_vali, y_train, y_vali = train_test_split(
    imgs, targets, test_size=0.3, random_state=42,shuffle = True)

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules

initial_learning_rate = 0.001
decay_steps = 1.0
decay_rate = 0.5
lr = optimizers.schedules.InverseTimeDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None)

from keras.callbacks import EarlyStopping,ReduceLROnPlateau
model.compile(optimizer=optimizers.Adam(learning_rate = 0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=3)
callbacks = [
#     EarlyStopping(monitor='val_loss',
#                   verbose=2,
#                   restore_best_weights=True,
#                   patience = 5
#                  ),
    reduce_lr
]

epochs = 50
batches = 128

history = model.fit(
    x = X_train,
    y = y_train,
    epochs=epochs,
    batch_size = batches,
    callbacks = callbacks,
    validation_data=(X_vali,y_vali),
    verbose = 1
)

# KERAS_MODEL_NAME = "flower_classiciation_model.h5"
# keras.models.save_model(model, KERAS_MODEL_NAME)

keras.models.load_model("flower_classiciation_model.h5")

filepath = r"C:\Users\ASUS\桌面\Xcelerate\CNN\flower color images\Flower_Classification\flower_images\flower_labels.csv"
test_ds = pd.read_csv(filepath)
test_ds.head(10)

def loadTestImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    targets = []
    for image in imagesList:
            img = np.array(Image.open(path+image).convert('RGB'))
            img = cv2.resize(img,(128,128))
            loadedImages.append(img)
    return np.array(loadedImages)

testpath = r"C:\Users\ASUS\桌面\Xcelerate\CNN\flower color images\Flower_Classification\flower_images\flower_images\\"

# your images in an array
test_imgs= loadTestImages(testpath)
test_targets = np.array(test_ds['label'])

predictions = model.predict(test_imgs)

y_pred = [np.argmax(pred) for pred in predictions]
y_test = test_targets

leng = len(y_pred)

label_dict = {
    0:"phlox",
1 : "rose",
2 : "calendula",
3 : "iris",
4 : "leucanthemum maximum (Shasta daisy)",
5 : "campanula (bellflower)",
6 : "viola",
7 : "rudbeckia laciniata (Goldquelle)",
8 : "peony",
9 : "aquilegia",
10: "rhododendron" ,
11 : "passiflora",
12 : "tulip",
13 : "water lily",
14 : "lilium",
15 : "veronica chamaedrys",
16 : "cosmos",
17 : "aster annual",
18 : "aster perennial",
19 : "snowdrop"
}

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=list(label_dict.keys())))

pred_label = [label_dict[i] for i in y_pred]

i = 0
for img in test_imgs[:20]:

    plt.title(pred_label[y_pred[i]])
    plt.imshow(img)
    plt.show()
    i += 1