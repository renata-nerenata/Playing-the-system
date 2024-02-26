import logging
import tensorflow as tf
import argparse
import random

from src.metrics.metrics import define_custom_loss
from src.data.make_puzzle import PuzzleInference

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
tf.get_logger().setLevel(logging.WARNING)

def main(model, puzzle, accepted_pairs, base_num_step):
    num_step = base_num_step + random.randint(-2, 2)

    if model == 'FCN':
        define_custom_loss()

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, None, None)),  # Adjust the input shape as needed
            tf.keras.layers.TFSMLayer("models/FCN", call_endpoint='serving_default')
        ])

        try:
            logging.info("Initializing Puzzle Inference...")
            puzzle_inference = PuzzleInference(puzzle, accepted_pairs, num_step)

            logging.info("Preprocessing tensor...")
            prep = puzzle_inference.preprocess_tensor()

            logging.info("Performing prediction...")
            puzzle_inference.pred_matrix = model.predict(prep)
            print(puzzle_inference.pred_matrix)

            puzzle_solution = puzzle_inference.insert_gap_at_indexes()

            print(puzzle_solution)
            score = puzzle_inference._gearbox_score_opt(puzzle_solution)
            logging.info(f"Gearbox Score: {score}")

            return puzzle_solution

        except Exception as e:
            logging.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Puzzle Inference Arguments")
    parser.add_argument(
        "--model",
        nargs="+",
        default='FCN',
        help="Name of a model",
    )
    parser.add_argument(
        "--puzzle",
        nargs="+",
        default=["CCGAG", "GTGAG", "CCGCG", "CCAGG", "CAGAGC", "GAGAG"],
        help="List of puzzle sequences",
    )
    parser.add_argument(
        "--accepted_pairs",
        nargs="+",
        default=[
                ("T", "G"),
                ("G", "C"),
                ("C", "T"),
                ("A", "G"),
                ("A", "C"),
                ("G", "A"),
                ("C", "T"),
                ("G", "T"),
            ],
        help="List of accepted pairs",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of steps (default: 4)",
    )
    args = parser.parse_args()
    main(args.model, args.puzzle, args.accepted_pairs, args.steps)

