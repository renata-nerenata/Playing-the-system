import numpy as np
import tensorflow as tf
import logging
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Reshape, Dense
from transformers import TFAutoModel


class AxialAttention(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(AxialAttention, self).__init__()
        self.dim = dim

    def build(self, input_shape):
        _, self.height, self.width, self.channels = input_shape
        self.query = self.add_weight(
            "query",
            shape=[1, 1, self.channels, self.dim],
            initializer="random_normal",
            trainable=True,
        )
        self.key = self.add_weight(
            "key",
            shape=[1, 1, self.channels, self.dim],
            initializer="random_normal",
            trainable=True,
        )
        self.value = self.add_weight(
            "value",
            shape=[1, 1, self.channels, self.dim],
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        query = tf.nn.conv2d(inputs, self.query, strides=[1, 1, 1, 1], padding="SAME")
        key = tf.nn.conv2d(inputs, self.key, strides=[1, 1, 1, 1], padding="SAME")
        value = tf.nn.conv2d(inputs, self.value, strides=[1, 1, 1, 1], padding="SAME")

        attention_scores = tf.nn.softmax(
            tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])), axis=-1
        )
        attended_value = tf.matmul(attention_scores, value)

        output = tf.nn.conv2d_transpose(
            attended_value, self.value, strides=[1, 1, 1, 1], padding="SAME"
        )
        return output


class TransformerModel:
    def __init__(self, latent_dim):
        """
        Args:
            latent_dim (int): Number of units in the transformer layer.
        """
        self.latent_dim = latent_dim
        self.model = None
        self.logger = logging.getLogger(__name__)

    def build_model(self):
        encoder_inputs = Input(shape=(25, 25, 4))
        reshaped_inputs = Reshape((25 * 25, 4))(encoder_inputs)

        transformer_layer = TFAutoModel.from_pretrained("transformer_model")
        transformer_outputs = transformer_layer(reshaped_inputs)

        decoder_inputs = RepeatVector(25 * 25)(transformer_outputs)
        transformer_decoder = TFAutoModel.from_pretrained("transformer_model")
        decoder_outputs = transformer_decoder(decoder_inputs)

        dense_layer = Dense(4)
        decoder_outputs = dense_layer(decoder_outputs)

        self.model = Model(encoder_inputs, decoder_outputs)
        self.model.compile(optimizer="adam", loss="mse")

    def train(self, input_data, target_data, epochs, batch_size):
        """
        Args:
            input_data (ndarray): Input data for training.
            target_data (ndarray): Target data for training.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        reshaped_target_data = np.reshape(
            target_data, (target_data.shape[0], 25 * 25, 4)
        )

        self.logger.info("Training started...")
        self.model.fit(
            input_data,
            reshaped_target_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        self.logger.info("Training completed.")

    def predict(self, input_data):
        """
        Args:
            input_data (ndarray): Input data for prediction.

        Returns:
            ndarray: Predicted output data.
        """
        return self.model.predict(input_data)


# TODO: add axial attention
class EncoderDecoderModel(tf.keras.Model):
    def __init__(self):
        super(EncoderDecoderModel, self).__init__()
        self.encoder_axial = AxialAttention(dim=8)
        self.decoder_axial = AxialAttention(dim=8)
        self.encoder_conv = tf.keras.layers.Conv2D(
            16, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.decoder_conv = tf.keras.layers.Conv2D(
            4, kernel_size=(1, 1), padding="same", activation="sigmoid"
        )

    def call(self, inputs):
        # Encoder
        encoded = self.encoder_axial(inputs)
        encoded = self.encoder_conv(encoded)

        # Decoder
        decoded = self.decoder_axial(encoded)
        decoded = self.decoder_conv(decoded)

        return decoded


def run_transformer(input_data, target_data):
    parser = argparse.ArgumentParser(description="Transformer Autoencoder")
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Number of units in the transformer layer",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    model = TransformerModel(args.latent_dim)
    model.build_model()
    model.train(input_data, target_data, epochs=args.epochs, batch_size=args.batch_size)
    predictions = model.predict(input_data)
    return predictions
