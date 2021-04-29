from lab5.ConvolutionModel import N_EPOCHS
from datetime import datetime
import matplotlib.pyplot as plt


def compare_results(model_raw_history, model_aug_history, name=False):
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

    ax[0].plot(epochs, aug_val_accuracy,
               label='val_accuracy - augmented', color='#fff000')
    ax[0].plot(epochs, aug_accuracy,
               label='accuracy - augmented', color='#ff4400')
    ax[0].plot(epochs, raw_val_accuracy,
               label='val_accuracy - raw', color='#000fff')
    ax[0].plot(epochs, raw_accuracy,
               label='accuracy - raw', color='#ff44ee')

    ax[0].set_title('Accuracy')
    ax[0].legend()
    ax[0].grid(lw=.2)

    ax[1].plot(epochs, aug_val_loss,
               label='val_loss - augmented', color='#fff000')
    ax[1].plot(epochs, aug_loss,
               label='loss - augmented', color='#ff4400')
    ax[1].plot(epochs, raw_val_loss,
               label='val_loss - raw', color='#000fff')
    ax[1].plot(epochs, raw_loss,
               label='loss - raw', color='#ff44ee')

    ax[1].set_title('Loss')
    ax[1].legend()
    ax[1].grid(lw=.2)

    if name:
        plt.savefig('lab5_results_{name}.png'.format(name=name), dpi=600)
    else:
        plt.savefig('lab5_results_{time}.png'.format(time=datetime.now()), dpi=600)
    plt.show()
