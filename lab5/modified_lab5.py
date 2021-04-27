
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator, random_shear, random_rotation, random_zoom, random_shift, random_brightness
from keras.callbacks import ModelCheckpoint

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

VERBOSE = True
N_CLASSES = 10
N_EPOCHS = 50
BATCH_SIZE = 50

TRAIN_SAMPLES = 60000  # of 60000
TEST_SAMPLES = 10000  # of 10000

raw_train, raw_test = keras.datasets.fashion_mnist.load_data()

raw_train_x, raw_train_y = raw_train
raw_test_x, raw_test_y = raw_test

# DATASET LIMIT NUMBER OF SAMPLES

raw_train_x = raw_train_x[:TRAIN_SAMPLES]
raw_train_y = raw_train_y[:TRAIN_SAMPLES]
raw_test_x = raw_test_x[:TEST_SAMPLES]
raw_test_y = raw_test_y[:TEST_SAMPLES]


# DATASET DETAILS

print(f'DATA INITIAL:')
print(f'TRAIN: {raw_train_x.shape} images + {raw_train_y.shape} labels')
print(f'TEST: {raw_test_x.shape} images + {raw_test_y.shape} labels')

# DATASET NORMALIZATION

raw_train_x = np.asarray(raw_train_x / 255., dtype=float)
raw_train_y = np.asarray(raw_train_y, dtype=np.int32)

raw_test_x = np.asarray(raw_test_x / 255., dtype=float)
raw_test_y = np.asarray(raw_test_y, dtype=np.int32)

# NN MODEL(S)


def build_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPool2D(strides=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(lr=1e-3),
        metrics=["accuracy"]
    )

    return model


model_raw = build_model()
model_aug = build_model()

if VERBOSE:
    print(model_raw.summary())

# SAMPLING


def raw_train_sample(samples_per_class=100):
    xs = []
    ys = []
    for label_i in range(N_CLASSES):
        label_i_xs = raw_train_x[raw_train_y == label_i]
        label_i_xs_sample = label_i_xs[np.random.randint(
            0, high=label_i_xs.shape[0], size=samples_per_class), :]
        xs.extend(label_i_xs_sample)
        ys.append([label_i] * samples_per_class)

    return np.array(xs, dtype=float), np.array(ys, dtype=np.int32)


def tensor_reshape_xs(xs):
    return np.array(xs).reshape(-1, 28, 28, 1)


def categorical_reshape_ys(ys):
    return to_categorical(ys)


# DATASET RESHAPING FOR TRAINING

raw_train_x_reshaped = tensor_reshape_xs(raw_train_x)
raw_train_y_reshaped = categorical_reshape_ys(raw_train_y)

raw_test_x_reshaped = tensor_reshape_xs(raw_test_x)
raw_test_y_reshaped = categorical_reshape_ys(raw_test_y)

print(f'DATA RESHAPED:')
print(
    f'TRAIN - XS: {raw_train_x_reshaped.shape} YS: {raw_train_y_reshaped.shape}')
print(
    f'TEST - XS: {raw_test_x_reshaped.shape} YS: {raw_test_y_reshaped.shape}')

# RAW MODEL TRAINING

model_raw_history = model_raw.fit(
    raw_train_x_reshaped,
    raw_train_y_reshaped,
    epochs=N_EPOCHS,
    batch_size=50,
    verbose=1,
    validation_data=(raw_test_x_reshaped, raw_test_y_reshaped),
    shuffle=True,
    callbacks=[
        ModelCheckpoint(
            'model_raw.json',
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch',
        )
    ]
)

if VERBOSE:
    print('RAW TRAINING FINAL RESULTS:')
    for key, values in model_raw_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

# DATASET IMAGE AUGMENTATION


def dataset_augmented(train_xs, train_ys, augmentation_loops=1):
    augmented_xs = []
    augmented_ys = []

    for i_x in range(0, train_xs.shape[0]):
        for i_loop in range(0, augmentation_loops):
            augmented_xs.append(train_xs[i_x])  # append original image too
            augmented_ys.append(train_ys[i_x])

            augmented_xs.append(random_rotation(train_xs[i_x], 20, row_axis=0, col_axis=1, channel_axis=2))
            augmented_ys.append(train_ys[i_x])

            augmented_xs.append(random_shear(train_xs[i_x], 0.2, row_axis=0, col_axis=1, channel_axis=2))
            augmented_ys.append(train_ys[i_x])

            augmented_xs.append(random_shift(train_xs[i_x], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
            augmented_ys.append(train_ys[i_x])

            augmented_xs.append(random_zoom(train_xs[i_x], (0.9, 1.1), row_axis=0, col_axis=1, channel_axis=2))
            augmented_ys.append(train_ys[i_x])

    return np.array(augmented_xs), np.array(augmented_ys)


aug_train_x, aug_train_y = dataset_augmented(
    raw_train_x_reshaped, raw_train_y_reshaped)
aug_test_x = raw_test_x_reshaped
aug_test_y = raw_test_y_reshaped

# AUGMENTED MODEL TRAINING

model_aug_history = model_aug.fit(
    aug_train_x,
    aug_train_y,
    epochs=N_EPOCHS,
    batch_size=50,
    verbose=1,
    validation_data=(aug_test_x, aug_test_y),
    shuffle=True,
    callbacks=[
        ModelCheckpoint(
            'model_aug.json',
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch',
        )
    ]
)

if VERBOSE:
    print('AUGMENTED TRAINING FINAL RESULTS:')
    for key, values in model_aug_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

# PLOTTING BOTH TRAINING RESULTS

epochs = list(range(N_EPOCHS))

aug_val_accuracy = model_aug_history.history['val_accuracy']
aug_accuracy = model_aug_history.history['accuracy']
aug_val_loss = model_aug_history.history['val_loss']
aug_loss = model_aug_history.history['loss']

raw_val_accuracy = model_raw_history.history['val_accuracy']
raw_accuracy = model_raw_history.history['accuracy']
raw_val_loss = model_raw_history.history['val_loss']
raw_loss = model_raw_history.history['loss']

fig, ax = plt.subplots(nrows=1, ncols=2)


l1 = ax[0].plot(epochs, aug_val_accuracy,
           label='val_accuracy - augmented', color='#fff000')
l2 = ax[0].plot(epochs, aug_accuracy,
           label='accuracy - augmented', color='#ff4400')
l3 = ax[0].plot(epochs, raw_val_accuracy,
           label='val_accuracy - raw', color='#000fff')
l4 = ax[0].plot(epochs, raw_accuracy,
           label='accuracy - raw', color='#ff44ee')

ax[0].set_title('Accuracy')
ax[0].legend()
ax[0].grid(lw=.2)

a1 = ax[1].plot(epochs, aug_val_loss,
           label='val_loss - augmented', color='#fff000')
a2 = ax[1].plot(epochs, aug_loss,
           label='loss - augmented', color='#ff4400')
a3 = ax[1].plot(epochs, raw_val_loss,
           label='val_loss - raw', color='#000fff')
a4 = ax[1].plot(epochs, raw_loss,
           label='loss - raw', color='#ff44ee')

ax[1].set_title('Loss')
ax[1].legend()
ax[1].grid(lw=.2)

plt.savefig('lab5_results.png', dpi=600)
plt.show()

print("DONE!")
