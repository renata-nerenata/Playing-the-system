import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.constants import DICT_SYMBOLS, STANDARD_ORDER


class Puzzle:
    def __init__(self, row):
        self.originalCode = row["originalCode"]
        self.nGaps = row["nGaps"]
        self.score = row["score"]
        self.playerSolutions = row["playerSolutions"]
        self.playerIDs = row["playerIDs"]
        self.steps = row["steps"]
        self.count_steps = row["count_steps"]
        self.accepted_pairs = row["accepted_pairs"]
        self.difficulty = row["difficulty"]
        self.max_gap_count = row["max_gap_count"]
        self.label = row["label"]
        self.puzzle = row["original"]

        self.calculate_original_and_solution()
        self.calculate_original_and_solution_encoded()

    def calculate_original_and_solution(self):
        self.original = self.build_puzzle_to_end(self.puzzle)
        self.solution = self.build_puzzle_to_end(self.playerSolutions)

    def calculate_original_and_solution_encoded(self):
        self.original_tensor = self.get_tensor_encoding_input(
            self.get_puzzle_with_consensus(self.original)
        )
        self.solution_tensor = self.get_tensor_encoding_input(
            self.get_puzzle_with_consensus(self.solution)
        )

    def build_puzzle_to_end(self, puzzle):
        accepted_pairs_len = len(self.accepted_pairs)
        return [
            row + "-" * (accepted_pairs_len - len(row))
            if len(row) < accepted_pairs_len
            else row
            for row in puzzle
        ]

    def plot_one_puzzle(self, puzzle):
        num_puzzle = np.array([[DICT_SYMBOLS[i] for i in row] for row in puzzle])
        rot_num_puzzle = np.rot90(num_puzzle, 1)
        rot_labels = np.rot90(np.array(puzzle), 1)

        fig = sns.heatmap(
            rot_num_puzzle, annot=rot_labels, fmt="", cmap="Pastel1_r", cbar=False
        )
        plt.axis("off")
        plt.show()
        return fig

    def gearbox_score(self, puzzle, bonus=1.15):
        consensus = self.accepted_pairs
        seqs = puzzle
        score = 0
        for col_ind in range(0, len(seqs[0])):
            col_bonus = True
            col_tot = 0
            for row in seqs:
                i = row[col_ind]
                if i == "-":
                    col_bonus = False
                    continue
                if i in consensus[col_ind]:
                    col_tot += 1
                else:
                    col_bonus = False

            if col_bonus:
                score += col_tot * bonus
            else:
                score += col_tot
        return score

    def get_tensor_encoding_input(self, puzzle, elements=STANDARD_ORDER):
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

    def get_puzzle_with_consensus(self, puzzle):
        first_string = "".join(pair[0] for pair in self.accepted_pairs)
        second_string = "".join(pair[1] for pair in self.accepted_pairs)

        full_puzzle = [first_string] + [second_string] + puzzle
        return full_puzzle


# TODO: Add decoding from tensor
