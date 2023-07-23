import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
import numpy as np


tf.get_logger().setLevel(logging.WARNING)


class FCN(Model):
    def __init__(self, dim1, dim2, channels):
        super(FCN, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.channels = channels
        self.build_model()

    def build_model(self):
        logging.info("Building the model...")
        inputs = Input((self.dim1, self.dim2, self.channels))

        x = Conv2D(16, (2, 2), activation="relu")(inputs)
        x = Conv2D(64, (2, 2), activation="relu")(x)
        x = Conv2D(128, (2, 2), activation="relu")(x)
        x = Conv2D(256, (2, 2), activation="relu")(x)

        x = Conv2DTranspose(128, (2, 2), strides=(1, 1))(x)
        x = Conv2DTranspose(64, (2, 2), strides=(1, 1))(x)
        x = Conv2DTranspose(32, (2, 2), strides=(1, 1))(x)
        x = Conv2DTranspose(16, (2, 2), strides=(1, 1))(x)

        output_cosine = Conv2D(1, (1, 1), activation="sigmoid", name="output_cosine")(x)
        output_mse = Conv2D(1, (1, 1), activation="sigmoid", name="output_mse")(x)

        self.model = Model(inputs=[inputs], outputs=[output_cosine, output_mse])
        self.model.compile(
            optimizer="adam",
            loss={
                "output_cosine": "cosine_similarity_loss",
                "output_mse": "mean_squared_error_loss",
            },
        )

    def summary(self):
        self.model.summary()


class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, training_data, target_data, epochs):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        total_start_time = time.time()
        avg_epoch_time = 0.0

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            logging.info(f"Epoch {epoch}/{epochs}")
            for i, (training_sample, target_sample) in enumerate(
                zip(training_data, target_data), 1
            ):
                logging.info(f"Training batch {i}/{len(training_data)}")
                self.model.train_on_batch(training_sample, target_sample)

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            avg_epoch_time = (avg_epoch_time * (epoch - 1) + epoch_time) / epoch

            remaining_epochs = epochs - epoch
            estimated_total_time = avg_epoch_time * remaining_epochs
            logging.info(f"Epoch time: {epoch_time:.2f} seconds")
            logging.info(
                f"Estimated remaining time: {estimated_total_time:.2f} seconds"
            )

        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time
        logging.info(f"Total training time: {total_training_time:.2f} seconds")
        logging.info("Training completed.")


class ModelEvaluator:
    def __init__(self, model, training_data, target_data):
        self.model = model
        self.training_data = training_data
        self.target_data = target_data

    def evaluate(self):
        loss_list = []

        for i, (training_sample, target_sample) in enumerate(
            zip(self.training_data, self.target_data), 1
        ):
            logging.info(f"Evaluating sample {i}/{len(self.training_data)}")
            loss = self.model.evaluate(training_sample, target_sample)
            loss_list.append(loss)

        return loss_list


def preprocess_data(df):
    training_data = []
    target_data = []

    for original, solution in zip(df["original"], df["solution"]):
        tr = np.moveaxis(np.array([original]).astype(float), 1, -1)
        solution = np.expand_dims(solution, axis=0)
        ta = np.moveaxis(np.array([solution]).astype(float), 1, -1)

        tr = np.where((tr == 0), 1.0, 0.0)
        ta = np.where((ta == 0), 1.0, 0.0)

        training_data.append(tr)
        target_data.append(ta)

    return training_data, target_data


def run_fcn(df):
    dim1 = None
    dim2 = None
    channels = 4

    training_data, target_data = preprocess_data(df)

    custom_model = FCN(dim1, dim2, channels)
    custom_model.summary()

    with tf.device("GPU:0"):
        model_trainer = ModelTrainer(custom_model.model)
        model_trainer.train(training_data, target_data, epochs=100)

    with tf.device("GPU:0"):
        model_evaluator = ModelEvaluator(custom_model.model, training_data, target_data)
        loss_list = model_evaluator.evaluate()
        print(loss_list)
