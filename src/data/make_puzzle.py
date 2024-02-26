import heapq
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from src.constants import STANDARD_ORDER, DICT_SYMBOLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
tf.get_logger().setLevel(logging.WARNING)

class PuzzleInference:
    def __init__(self, puzzle, accepted_pairs, n_gaps):
        self.original_tensor = None
        self.original = None
        self.n_gaps = n_gaps
        self.accepted_pairs = accepted_pairs
        self.puzzle = puzzle
        self.pred_matrix = None
        self._calculate_original_and_solution()

    def _calculate_original_and_solution(self):
        self.original = self._build_puzzle_to_end(self.puzzle)
        self.original_tensor = self._get_tensor_encoding_input(
            self._get_puzzle_with_consensus(self.original)
        )

    def _build_puzzle_to_end(self, puzzle):
        accepted_pairs_len = len(self.accepted_pairs)
        return [
            row + "-" * (accepted_pairs_len - len(row))
            if len(row) < accepted_pairs_len
            else row
            for row in puzzle
        ]

    def _plot_one_puzzle(self, puzzle):
        puzzle_as_nums = np.array([[DICT_SYMBOLS[i] for i in row] for row in puzzle])
        rot_puzzle_as_nums = np.rot90(puzzle_as_nums, 1)
        labels = np.rot90(np.array(puzzle), 1)
        fig = sns.heatmap(
            rot_puzzle_as_nums, annot=labels, fmt="", cmap="Pastel1_r", cbar=False
        )
        plt.axis("off")
        plt.show()
        return fig

    def _transpose_puzzle(self, puzzle):
        string_length = len(puzzle[0])
        if all(len(string) == string_length for string in puzzle):
            transposed_list = ["".join(row) for row in zip(*puzzle)]
        return transposed_list
        # return list(len(string) == string_length for string in puzzle)

    def _gearbox_score_opt(self, puzzle):
        transposed_list = self._transpose_puzzle(puzzle)
        sum_total = 0
        for index, pair_nuc in enumerate(self.accepted_pairs):
            sum_line = [el in list(pair_nuc) for el in transposed_list[index]].count(
                True
            )
            if sum_line == len(transposed_list[index]):
                sum_line = sum_line * 1.15
            sum_total = sum_total + sum_line
        return sum_total

    @staticmethod
    def _get_tensor_encoding_input(puzzle, elements=STANDARD_ORDER):
        max_len = max(len(row) for row in puzzle)
        tensor = []
        for element in elements:
            element_tensor = [
                [1 if char == element else 0 for char in row]
                + [0] * (max_len - len(row))
                for row in puzzle
            ]
            tensor.append(element_tensor)
        return tensor

    def _get_puzzle_with_consensus(self, puzzle):
        first_string = "".join(pair[0] for pair in self.accepted_pairs)
        second_string = "".join(pair[1] for pair in self.accepted_pairs)
        full_puzzle = [first_string] + [second_string] + puzzle
        return full_puzzle

    def preprocess_tensor(self):
        tr = np.moveaxis(np.array([self.original_tensor]).astype(float), 1, -1)
        tr = np.where((tr == 0), 1.0, 0.0)
        return tr

    def restore_prediction_tensor(self):
        if self.pred_matrix is not None and 'output_mse' in self.pred_matrix:
            pred_matrix_array = self.pred_matrix['output_mse']
            if pred_matrix_array.ndim >= 3:
                i_restored = np.moveaxis(pred_matrix_array, -1, 1)
                reshaped_matrix = i_restored.flatten()
                min_elements = sorted(heapq.nsmallest(self.n_gaps, reshaped_matrix))
                result = np.where(np.isin(i_restored, min_elements), 1, 0)
                return result[0][0][2:]
            else:
                logging.error("pred_matrix_array does not have the expected dimensions.")
                return np.array([])
        else:
            logging.error("'output_mse' key not found in pred_matrix.")
            return np.array([])

    def find_indexes_of_ones_tensor(self):
        indexes = tf.where(tf.equal(self.restore_prediction_tensor(), 1))
        return indexes

    def insert_gap_at_indexes(self):
        sequence = self.original
        indexes = self.find_indexes_of_ones_tensor()
        result = list(sequence)  # Convert the sequence to a list to make modifications easier
        for index in indexes:
            row, col = index
            result[row] = result[row][:col] + '-' + result[row][col:len(sequence[0])-1]
        return result


