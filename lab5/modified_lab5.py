import warnings

import keras
import numpy as np
from keras.utils import to_categorical

from lab5.ConvolutionModel import ConvolutionModel
from lab5.augmentations import dataset_augmentation_mixed, dataset_augmentation_rotations, dataset_augmentation_shifts, dataset_augmentation_random, dataset_augmentation_rotations_shifts
from lab5.plotting import compare_results

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

VERBOSE = True
N_CLASSES = 10
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

model_raw = ConvolutionModel(raw_train_x_reshaped, raw_train_y_reshaped, raw_test_x_reshaped, raw_test_y_reshaped)
model_raw_history = model_raw.train()

if VERBOSE:
    print('RAW TRAINING FINAL RESULTS:')
    for key, values in model_raw_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

# TEST DATA FOR AUGMENTED RESULTS

aug_test_x = raw_test_x_reshaped
aug_test_y = raw_test_y_reshaped

# ROTATION AUGMENTATION

aug_train_x, aug_train_y = dataset_augmentation_rotations(raw_train_x_reshaped, raw_train_y_reshaped)

# AUGMENTED MODEL TRAINING

model_aug_rotations = ConvolutionModel(aug_train_x, aug_train_y, aug_test_x, aug_test_y)
model_aug_rotations_history = model_aug_rotations.train()

if VERBOSE:
    print('AUGMENTED TRAINING FINAL RESULTS:')
    for key, values in model_aug_rotations_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

compare_results(model_raw_history, model_aug_rotations_history, "rotations")

# SHIFTS AUGMENTATION

aug_train_x, aug_train_y = dataset_augmentation_shifts(raw_train_x_reshaped, raw_train_y_reshaped)

# AUGMENTED MODEL TRAINING

model_aug_rotations = ConvolutionModel(aug_train_x, aug_train_y, aug_test_x, aug_test_y)
model_aug_rotations_history = model_aug_rotations.train()

if VERBOSE:
    print('AUGMENTED TRAINING FINAL RESULTS:')
    for key, values in model_aug_rotations_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

compare_results(model_raw_history, model_aug_rotations_history, "shifts")

# SHIFTS AND ROTATIONS AUGMENTATION

aug_train_x, aug_train_y = dataset_augmentation_rotations_shifts(raw_train_x_reshaped, raw_train_y_reshaped)

# AUGMENTED MODEL TRAINING

model_aug_rotations = ConvolutionModel(aug_train_x, aug_train_y, aug_test_x, aug_test_y)
model_aug_rotations_history = model_aug_rotations.train()

if VERBOSE:
    print('AUGMENTED TRAINING FINAL RESULTS:')
    for key, values in model_aug_rotations_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

compare_results(model_raw_history, model_aug_rotations_history, "shifts_and_rotations")

# RANDOM AUGMENTATION - ONE LOOP

aug_train_x, aug_train_y = dataset_augmentation_random(raw_train_x_reshaped, raw_train_y_reshaped, 1)

# AUGMENTED MODEL TRAINING

model_aug_rotations = ConvolutionModel(aug_train_x, aug_train_y, aug_test_x, aug_test_y)
model_aug_rotations_history = model_aug_rotations.train()

if VERBOSE:
    print('AUGMENTED TRAINING FINAL RESULTS:')
    for key, values in model_aug_rotations_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

compare_results(model_raw_history, model_aug_rotations_history, "random_50_percent")

# RANDOM AUGMENTATION - TWO LOOPS

aug_train_x, aug_train_y = dataset_augmentation_random(raw_train_x_reshaped, raw_train_y_reshaped, 2)

# AUGMENTED MODEL TRAINING

model_aug_rotations = ConvolutionModel(aug_train_x, aug_train_y, aug_test_x, aug_test_y)
model_aug_rotations_history = model_aug_rotations.train()

if VERBOSE:
    print('AUGMENTED TRAINING FINAL RESULTS:')
    for key, values in model_aug_rotations_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

compare_results(model_raw_history, model_aug_rotations_history, "random_100_percent")

# RANDOM AUGMENTATION - THREE LOOPS

aug_train_x, aug_train_y = dataset_augmentation_random(raw_train_x_reshaped, raw_train_y_reshaped, 3)

# AUGMENTED MODEL TRAINING

model_aug_rotations = ConvolutionModel(aug_train_x, aug_train_y, aug_test_x, aug_test_y)
model_aug_rotations_history = model_aug_rotations.train()

if VERBOSE:
    print('AUGMENTED TRAINING FINAL RESULTS:')
    for key, values in model_aug_rotations_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

compare_results(model_raw_history, model_aug_rotations_history, "random_150_percent")

# RANDOM AUGMENTATION - FOUR LOOPS

aug_train_x, aug_train_y = dataset_augmentation_random(raw_train_x_reshaped, raw_train_y_reshaped, 4)

# AUGMENTED MODEL TRAINING

model_aug_rotations = ConvolutionModel(aug_train_x, aug_train_y, aug_test_x, aug_test_y)
model_aug_rotations_history = model_aug_rotations.train()

if VERBOSE:
    print('AUGMENTED TRAINING FINAL RESULTS:')
    for key, values in model_aug_rotations_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

compare_results(model_raw_history, model_aug_rotations_history, "random_200_percent")

# MIXED AUGMENTATION

aug_train_x, aug_train_y = dataset_augmentation_mixed(raw_train_x_reshaped, raw_train_y_reshaped)

# AUGMENTED MODEL TRAINING

model_aug_mixed = ConvolutionModel(aug_train_x, aug_train_y, aug_test_x, aug_test_y)
model_aug_mixed_history = model_aug_mixed.train()

if VERBOSE:
    print('AUGMENTED TRAINING FINAL RESULTS:')
    for key, values in model_aug_mixed_history.history.items():
        print(f'{key}: {values[-1]:.6f}')

compare_results(model_raw_history, model_aug_mixed_history)

print("DONE!")
