import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential

N_EPOCHS = 50


class ConvolutionModel:

    def __init__(self, train_x, train_y, test_x, test_y):
        self.model = self.build_model()
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    @staticmethod
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

    def train(self):
        return self.model.fit(
            self.train_x,
            self.train_y,
            epochs=N_EPOCHS,
            batch_size=50,
            verbose=1,
            validation_data=(self.test_x, self.test_y),
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
