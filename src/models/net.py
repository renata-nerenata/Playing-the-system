import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DataPreprocessor:
    @staticmethod
    def find_max_shape(dataset):
        max_shape = (0, 0, 0)
        for data in dataset:
            data_np = np.array(data)
            shape = data_np.shape
            if len(shape) == 3:
                max_shape = tuple(max(i, j) for i, j in zip(max_shape, shape))
        return max_shape

    @staticmethod
    def pad_data(dataset, max_shape):
        padded_dataset = []
        for data in dataset:
            data_np = np.array(data)
            padded_data = np.pad(
                data_np,
                [
                    (0, max_dim - cur_dim)
                    for max_dim, cur_dim in zip(max_shape, data_np.shape)
                ],
            )
            padded_dataset.append(padded_data)
        return padded_dataset

    def preprocess(self, input_data, target_data):
        max_shape_input = self.find_max_shape(input_data)
        max_shape_target = self.find_max_shape(target_data)
        input_data_padded = self.pad_data(input_data, max_shape_input)
        target_data_padded = self.pad_data(target_data, max_shape_target)

        input_data_tensors = [
            torch.tensor(data, dtype=torch.float32) for data in input_data_padded
        ]
        target_data_tensors = [
            torch.tensor(data, dtype=torch.float32) for data in target_data_padded
        ]

        return (
            input_data_tensors,
            target_data_tensors,
            max_shape_input,
            max_shape_target,
        )


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Trainer:
    def __init__(
        self,
        input_data,
        target_data,
        max_shape_input,
        max_shape_target,
        num_epochs,
        learning_rate,
    ):
        self.input_data = input_data
        self.target_data = target_data
        self.max_shape_input = max_shape_input
        self.max_shape_target = max_shape_target
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._initialize_model()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def _initialize_model(self):
        input_size = np.prod(self.max_shape_input)
        output_size = np.prod(self.max_shape_target)
        net = Net(input_size, output_size)
        return net.to(self.device)

    def train(self):
        for epoch in range(self.num_epochs):
            for i in range(len(self.input_data)):
                inputs = self.input_data[i].to(self.device)
                labels = self.target_data[i].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, torch.flatten(labels))
                loss.backward()
                self.optimizer.step()

                logging.info(
                    f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.input_data)}], Loss: {loss.item()}"
                )

    def predict(self, new_data):
        preprocessor = DataPreprocessor()
        new_data_padded = preprocessor.pad_data(new_data, self.max_shape_input)
        new_data_tensors = [
            torch.tensor(data, dtype=torch.float32).view(-1) for data in new_data_padded
        ]

        self.net.eval()
        predictions = []
        with torch.no_grad():
            for new_data_tensor in new_data_tensors:
                prediction = self.net(new_data_tensor.unsqueeze(0))
                predictions.append(prediction)

        predictions_np = [
            np.where(prediction.cpu().numpy() > 0.5, 1, 0) for prediction in predictions
        ]
        return predictions_np


def run_net(input_data, target_data):
    parser = argparse.ArgumentParser(description="Neural Network Trainer")
    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate for training"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    preprocessor = DataPreprocessor()
    (
        input_data_tensors,
        target_data_tensors,
        max_shape_input,
        max_shape_target,
    ) = preprocessor.preprocess(input_data, target_data)

    trainer = Trainer(
        input_data_tensors,
        target_data_tensors,
        max_shape_input,
        max_shape_target,
        args.epochs,
        args.lr,
    )
    trainer.train()

    # Assume new_data is the new input data for prediction
    new_data = data.matrix_x
    predictions = trainer.predict(new_data)
