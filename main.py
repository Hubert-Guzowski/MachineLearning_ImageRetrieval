import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Convolution2D, Conv2D, MaxPool2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import keras
from keras import layers
import time

# Required for computing average_precision_score
tf.compat.v1.enable_eager_execution()

TRAIN_PATH = './data/DataSet_SEQUENCE_1'
TEST_PATH = './data/DataSet_SEQUENCE_2'
MODEL_DIR = './data/'
MODEL_PATH = MODEL_DIR + 'saved_model'
IMAGE_DIM = (640, 480)
INPUT_DIM = IMAGE_DIM + (3,)
BATCH = 10
WORKERS = 10

def directory_dataset(d_path, subset):
    return image_dataset_from_directory(
        d_path,
        validation_split=0.2,
        subset=subset,
        seed=1337,
        image_size=IMAGE_DIM,
        batch_size=BATCH,
        labels='inferred',
        label_mode='categorical'
    )

def directory_sets(d_path):
    return directory_dataset(d_path, "training"), directory_dataset(d_path, "validation") 


print('Loading data...')

train_ds, val_ds = directory_sets(TRAIN_PATH)

print('Building model...')

import keras
from keras import layers


def get_augmentation():
    data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ])
    return data_augmentation


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    data_augmentation = get_augmentation()
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

model = make_model(INPUT_DIM, len(train_ds.class_names))
print(model.summary())

print('Training the model...')

EPOCHS = 10

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

def save_model_now():
    CHECKPOINT_PATH = MODEL_PATH + '_' + str(int(time.time()))
    print(f'Saving trained model to {CHECKPOINT_PATH}')
    model.save(CHECKPOINT_PATH)

save_model_now()

print('Loading test data...')

test_ds = image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_DIM,
    batch_size=BATCH,
    labels='inferred',
    label_mode='categorical'
)

print('Evaluating mAP...')

CLASSES_COUNT = len(test_ds.class_names)

predictions = model.predict(test_ds)
true_labels = np.concatenate([y for x, y in test_ds], axis=0)


APs = [average_precision_score(true_labels[:, k], predictions[:, k]) for k in range(0, CLASSES_COUNT)]
mAP = np.sum(APs) / CLASSES_COUNT

print(f'Mean average precision score (mAP): {mAP}')

