import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization


class FCNmodel:
    def __init__(self, input_data, target_data, learning_rate=0.001):
        self.input_data = input_data
        self.target_data = target_data
        self.learning_rate = learning_rate

    def create_encoder_decoder_model(self):
        input_shape = self.input_data.shape[1:]
        inputs = Input(shape=input_shape)

        encoder = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        encoder = BatchNormalization()(encoder)
        encoder = Conv2D(64, (3, 3), activation="relu", padding="same")(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.25)(encoder)

        decoder = Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(
            encoder
        )
        decoder = BatchNormalization()(decoder)
        decoder = Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(
            decoder
        )
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.25)(decoder)

        output = Conv2D(4, (3, 3), activation="sigmoid", padding="same")(decoder)

        model = Model(inputs=inputs, outputs=output)
        return model

    def train_model(self, epochs=20, batch_size=32):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error")

        # Apply additional data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
        )
        self.model.fit(
            datagen.flow(self.input_data, self.target_data, batch_size=batch_size),
            epochs=epochs,
        )

    def set_logging(self, log_file):
        logging.basicConfig(
            filename=log_file, level=logging.INFO, format="%(asctime)s %(message)s"
        )

    def predict_data(self, data):
        return self.model.predict(data)


def run_fcn(input_data, target_data):
    parser = argparse.ArgumentParser(description="Autoencoder")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--log_file", type=str, default="training.log", help="log file")
    args, _ = parser.parse_known_args()

    # Generate random input and target data
    input_data = np.round(np.random.rand(100, 5, 5, 4))
    target_data = np.round(np.random.rand(100, 5, 5, 4))

    autoencoder = FCNmodel(input_data, target_data, learning_rate=args.learning_rate)
    autoencoder.model = autoencoder.create_encoder_decoder_model()
    autoencoder.set_logging(args.log_file)
    autoencoder.train_model(epochs=args.epochs, batch_size=args.batch_size)

    # Generate predictions using the trained model
    test_data = np.round(np.random.rand(10, 5, 5, 4))  # Example test data
    predictions = autoencoder.predict_data(test_data)
    return predictions
