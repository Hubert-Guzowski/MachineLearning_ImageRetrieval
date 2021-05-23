import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras_pyramid_pooling_module import PyramidPoolingModule

from keras.optimizers import RMSprop

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

np.random.seed(222)


def plot_image_grid(data, figsize=(15, 15), cmap=None, cbar=True):
    """
    Plot the data as a grid of images.

    Args:
        data: the tensor of image data to plot in
        (M, N, H, W, C) format where M is the 
        height of the image grid, N is the width
        of the image grid, H is the height of the
        image, W is the width of the image, and C
        is the channel dimensionality of the image
        cmap: the color map to use for the data
        cbar: whether to include a color bar legend

    Returns:
        None

    """
    M, N = data.shape[0], data.shape[1]
    fig, ax = plt.subplots(nrows=M, ncols=N, sharex=True, sharey=True, figsize=figsize)
    for i in range(M):
        for j in range(N):
            idx = i + 1 + N * j
            im = ax[i, j].imshow(data[i, j], cmap=cmap)
            ax[i, j].axes.xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].axes.yaxis.set_major_locator(plt.NullLocator())
    if cbar:
        cb_ax = fig.add_axes([1., 0.2, 0.02, 0.6])
        cbar = fig.colorbar(im, cax=cb_ax)


# Dataset


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# normalize images into [0, 1]
X_train = X_train[..., None] / 255.0
X_test = X_test[..., None] / 255.0
# get the target size of the images and number of classes
TARGET_SIZE = X_train.shape[1:]
NUM_CLASSES = np.max(y_train) + 1
# convert discrete labels to one-hot vectors
y_train = np.eye(NUM_CLASSES)[y_train.flatten()]
y_test = np.eye(NUM_CLASSES)[y_test.flatten()]

print('X_train.shape={} y_train.shape={}'.format(X_train.shape, y_train.shape))
print('X_test.shape={} y_test.shape={}'.format(X_test.shape, y_test.shape))

plot_image_grid(X_train[:25].reshape(5, 5, 28, 28, 1), cbar=False, cmap='bone')
# plt.show()


input_layer = Input(TARGET_SIZE)
x = input_layer
x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(10, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=x)
rms = RMSprop()
model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

X_train_, y_train_ = X_train, y_train
num = 1000
X_train, y_train = [], []
X_train = np.array(X_train_)
y_train = np.array(y_train_)
X_train = X_train[:num, :, :, :]
y_train = y_train_[:num]

print('X_train.shape={} y_train.shape={}'.format(X_train.shape, y_train.shape))
print('X_test.shape={} y_test.shape={}'.format(X_test.shape, y_test.shape))

history1 = model.fit(X_train, y_train,
                     epochs=20,
                     batch_size=10,
                     validation_split=0.3,
                     shuffle=True,
                     )

df1 = pd.DataFrame(history1.history)
print(df1)

print(history1.history.keys())
print('\n')

# Plot training & validation accuracy values
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

loss1, accuracy1 = model.evaluate(X_test, y_test)
print('accuracy1={}'.format(accuracy1))
print('\n')

# =============================== PyramidPoolingModule() ===============================
# ## Pyramid Pooling Near Output

print('\n')
print('Pyramid Pooling Near Output')

input_layer = Input(TARGET_SIZE)
x = input_layer
x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = PyramidPoolingModule(1, (3, 3), padding='same')(x)
x = Flatten()(x)
x = Dense(10, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=x)
model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history2 = model.fit(X_train, y_train,
                     epochs=20,
                     batch_size=10,
                     validation_split=0.3,
                     shuffle=True,
                     )

df2 = pd.DataFrame(history2.history)
print(df2)

loss2, accuracy2 = model.evaluate(X_test, y_test)
print('accuracy2={}'.format(accuracy2))

# ## Pyramid Pooling Near Input
print('\n')
print('Pyramid Pooling Near Input')

input_layer = Input(TARGET_SIZE)
x = input_layer
x = PyramidPoolingModule(1, (3, 3), padding='same')(x)
x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(10, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=x)
model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history3 = model.fit(X_train, y_train,
                     epochs=20,
                     batch_size=10,
                     validation_split=0.3,
                     shuffle=True,
                     )

df3 = pd.DataFrame(history3.history)
print(df3)

loss3, accuracy3 = model.evaluate(X_test, y_test)
print('accuracy3={}'.format(accuracy3))

print('\n')
