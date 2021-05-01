import numpy as np
import random
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D
from keras.layers import MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Activation, Concatenate
from keras import optimizers
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.initializers import RandomUniform
from sklearn.model_selection import train_test_split

from IPython.display import SVG
from matplotlib import pyplot as plt
import math

import os

# CONSTANTS

N_CLASSES = 10
N_EPOCHS = 40

BATCH = 128

METRIC_THRESHOLD = .5

# UTILITY FUNCTIONS

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, _ = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(1 - y_pred, 0))
    return K.mean(y_true * square_pred + (1-y_true) * margin_square)


def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(N_CLASSES)]) - 1
    for d in range(N_CLASSES):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]  # positive sample
            inc = random.randrange(1, N_CLASSES)
            dn = (d + inc) % N_CLASSES
            # negative sample
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels, dtype=np.float32)


def create_pairs_per_class(x, digit_indices, class_1, class_2):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(N_CLASSES)]) - 1
    for i in range(n):
        z1, z2 = digit_indices[class_1][i], digit_indices[class_1][i + 1]
        pairs += [[x[z1], x[z2]]]  # positive sample
        # negative sample
        z1, z2 = digit_indices[class_1][i], digit_indices[class_2][i]
        pairs += [[x[z1], x[z2]]]
        labels += [1, 0]
    return np.array(pairs), np.array(labels, dtype=np.float32)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < METRIC_THRESHOLD
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < METRIC_THRESHOLD, y_true.dtype)))


# SIAMESE NEURAL NETWORK - SqueezeNet


def fire_module(x, s1x1, e1x1, e3x3, name):
    #Squeeze layer
    squeeze = Conv2D(s1x1, (1, 1), activation='relu', padding='valid',
                     kernel_initializer='glorot_uniform', name=name + 's1x1')(x)
    squeeze_bn = BatchNormalization(name=name+'sbn')(squeeze)

    #Expand 1x1 layer and 3x3 layer are parallel

    #Expand 1x1 layer
    expand1x1 = Conv2D(e1x1, (1, 1), activation='relu', padding='valid',
                       kernel_initializer='glorot_uniform', name=name + 'e1x1')(squeeze_bn)

    #Expand 3x3 layer
    expand3x3 = Conv2D(e3x3, (3, 3), activation='relu', padding='same',
                       kernel_initializer='glorot_uniform', name=name + 'e3x3')(squeeze_bn)

    #Concatenate expand1x1 and expand 3x3 at filters
    output = Concatenate(axis=3, name=name)([expand1x1, expand3x3])

    return output


def SqueezeNet(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(96, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='Conv1')(inputs)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Maxpool1')(conv1)
    batch1 = BatchNormalization(name='Batch1')(maxpool1)

    # fire2 = fire_module(batch1, 16, 64, 64, "Fire2")
    # fire3 = fire_module(fire2, 16, 64, 64, "Fire3")

    fire4 = fire_module(batch1, 32, 128, 128, "Fire2")
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Maxpool2')(fire4)

    # fire5 = fire_module(maxpool4, 32, 128, 128, "Fire5")
    
    fire6 = fire_module(maxpool4, 48, 192, 192, "Fire3")
    fire7 = fire_module(fire6, 48, 192, 192, "Fire4")
    fire8 = fire_module(fire7, 48, 192, 192, "Fire5")
    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Maxpool5')(fire8)

    # fire9 = fire_module(maxpool8, 64, 256, 256, "Fire9")
    
    dropout = Dropout(0.5, name="Dropout")(maxpool8)
    conv10 = Conv2D(10, kernel_size=(1, 1), strides=(1, 1),padding='same', activation='relu', name='Conv6')(dropout)
    batch10 = BatchNormalization(name='Batch6')(conv10)
    avgpool10 = GlobalAveragePooling2D(name='GlobalAvgPool6')(batch10)

    # softmax = Activation('softmax')(avgpool10)

    squeezenet = Model(inputs=inputs, outputs=avgpool10)
    return squeezenet


def create_squeeze_net(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    return SqueezeNet(input_shape)


# SIAMESE NEURAL NETWORK - Simple

def create_simple_siamese(input_shape):
    inputs = Input(input_shape)

    # first CONV => RELU => POOL => DROPOUT
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # second CONV => RELU => POOL => DROPOUT
    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # outputs
    pooled_output = GlobalAveragePooling2D()(x)
    outputs = Dense(48)(pooled_output)

    return Model(inputs, outputs)

# SIAMESE NEURAL NETWORK - Simple 2

def create_simple2_siamese(input_shape):
    inputs = Input(input_shape)
    
    x = Conv2D(4, (5,5), activation = 'tanh')(inputs)
    x = AveragePooling2D(pool_size = (2,2))(x)
    x = Conv2D(16, (5,5), activation = 'tanh')(x)
    x = AveragePooling2D(pool_size = (2,2))(x)
    x = Flatten()(x)
    outputs = Dense(10, activation = 'tanh')(x)
    return Model(inputs, outputs)

# ===============================================================================

# DATA PREPARATION

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]
print('INPUT SHAPE:', input_shape)


digit_indices = [np.where(y_train == i)[0] for i in range(N_CLASSES)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)


digit_indices = [np.where(y_test == i)[0] for i in range(N_CLASSES)]
te_pairs, te_y = create_pairs(x_test, digit_indices)


print("Shape of training pairs", tr_pairs.shape)
print("Shape of training labels", tr_y.shape)

# NETWORK DEFINITION

squeeze_net = create_squeeze_net(input_shape)
simple_net = create_simple_siamese(input_shape)
simple2_net = create_simple2_siamese(input_shape)

for (net_name, net_model) in [
    ('SimpleSiamese', simple_net), 
    ('SimpleSiamese2', simple2_net), 
    ('SqueezeNet', squeeze_net)]:
    input_a = Input(shape=input_shape, name='input_a')
    input_b = Input(shape=input_shape, name='input_b')

    # because we re-use the same instance `squeeze_net`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = net_model(input_a)
    processed_b = net_model(input_b)

    net_distance = Lambda(
        euclidean_distance,
        output_shape=eucl_dist_output_shape
    )([processed_a, processed_b])

    model = Model([input_a, input_b], net_distance)
    model.summary()

    # MODEL VISUALIZATION

    model_to_dot(
        net_model,
        show_layer_names=True,
        show_shapes=True
    ).write_png(f'base_{net_name}.png')

    model_to_dot(
        model,
        show_layer_names=True,
        show_shapes=True
    ).write_png(f'combined_{net_name}.png')

    # MODEL TRAINING

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    checkpointer = ModelCheckpoint(filepath=f'{net_name}.hfs5', save_best_only=True)
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                        batch_size=BATCH,
                        epochs=N_EPOCHS,
                        validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
                        callbacks=[checkpointer])


    # TRAINING PROGRESS PLOTTING

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'accuracy_{net_name}_{N_EPOCHS}.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'loss_{net_name}_{N_EPOCHS}.png')
    plt.show()

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)

    print(f'* {net_name} accuracy on training set: {100 * tr_acc:0.2f}')
    print(f'* {net_name} Accuracy on test set: {100 * te_acc:0.2f}')

    final_result = model.evaluate([te_pairs[:, 0], te_pairs[:, 1]], te_y)

    print(f'{net_name} result: {final_result}')

    # ???

    # model = load_model(f'{net_name}.hfs5', custom_objects={'contrastive_loss': contrastive_loss})
    # digit_indices = [np.where(y_test == i)[0] for i in range(N_CLASSES)]
    # for i in range(N_CLASSES):
    #     for j in range(N_CLASSES):
    #         print(i, j)
    #         te_pairs, te_y = create_pairs_per_class(x_test, digit_indices, i, j)
    #         result = model.evaluate([te_pairs[:, 0], te_pairs[:, 1]], te_y)
    #         print(result)
