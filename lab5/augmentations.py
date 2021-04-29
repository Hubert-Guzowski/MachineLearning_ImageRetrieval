import numpy as np
from random import random
from keras.preprocessing.image import random_shear, random_rotation, random_zoom, random_shift, random_brightness


def dataset_augmentation_rotations(train_xs, train_ys, augmentation_loops=1):
    augmented_xs = []
    augmented_ys = []

    for i_x in range(0, train_xs.shape[0]):
        for i_loop in range(0, augmentation_loops):
            augmented_xs.append(train_xs[i_x])  # append original image too
            augmented_ys.append(train_ys[i_x])

            augmented_xs.append(random_rotation(train_xs[i_x], 45, row_axis=0, col_axis=1, channel_axis=2))
            augmented_ys.append(train_ys[i_x])

    return np.array(augmented_xs), np.array(augmented_ys)


def dataset_augmentation_shifts(train_xs, train_ys, augmentation_loops=1):
    augmented_xs = []
    augmented_ys = []

    for i_x in range(0, train_xs.shape[0]):
        for i_loop in range(0, augmentation_loops):
            augmented_xs.append(train_xs[i_x])  # append original image too
            augmented_ys.append(train_ys[i_x])

            augmented_xs.append(random_shift(train_xs[i_x], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
            augmented_ys.append(train_ys[i_x])

    return np.array(augmented_xs), np.array(augmented_ys)


def dataset_augmentation_rotations_shifts(train_xs, train_ys, augmentation_loops=1):
    augmented_xs = []
    augmented_ys = []

    for i_x in range(0, train_xs.shape[0]):
        for i_loop in range(0, augmentation_loops):
            augmented_xs.append(train_xs[i_x])  # append original image too
            augmented_ys.append(train_ys[i_x])

            roulette_wheel_selection = random()

            if roulette_wheel_selection < 0.5:
                augmented_xs.append(random_rotation(train_xs[i_x], 45, row_axis=0, col_axis=1, channel_axis=2))
                augmented_ys.append(train_ys[i_x])
            else:
                augmented_xs.append(random_shift(train_xs[i_x], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
                augmented_ys.append(train_ys[i_x])

    return np.array(augmented_xs), np.array(augmented_ys)


def dataset_augmentation_random(train_xs, train_ys, augmentation_loops=1):
    augmented_xs = []
    augmented_ys = []

    for i_x in range(0, train_xs.shape[0]):
        augmented_xs.append(train_xs[i_x])  # append original image too
        augmented_ys.append(train_ys[i_x])

        for i_loop in range(0, augmentation_loops):
            roulette_wheel_selection = random()

            if roulette_wheel_selection < 0.1:
                augmented_xs.append(random_rotation(train_xs[i_x], 45, row_axis=0, col_axis=1, channel_axis=2))
                augmented_ys.append(train_ys[i_x])
            elif roulette_wheel_selection < 0.2:
                augmented_xs.append(random_shear(train_xs[i_x], 0.2, row_axis=0, col_axis=1, channel_axis=2))
                augmented_ys.append(train_ys[i_x])
            elif roulette_wheel_selection < 0.3:
                augmented_xs.append(random_shift(train_xs[i_x], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
                augmented_ys.append(train_ys[i_x])
            elif roulette_wheel_selection < 0.4:
                augmented_xs.append(random_zoom(train_xs[i_x], (0.9, 1.1), row_axis=0, col_axis=1, channel_axis=2))
                augmented_ys.append(train_ys[i_x])
            elif roulette_wheel_selection < 0.5:
                augmented_xs.append(random_brightness(train_xs[i_x], (0.5, 1.5)))
                augmented_ys.append(train_ys[i_x])

    return np.array(augmented_xs), np.array(augmented_ys)


def dataset_augmentation_mixed(train_xs, train_ys, augmentation_loops=1):
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
