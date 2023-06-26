import numpy as np
from scipy.spatial import distance

def get_cosine_similarity(distribution_1, distribution_2):
    a = np.matrix.flatten(distribution_1)
    b = np.matrix.flatten(distribution_2)
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_hamming_distance(a, b):
    return distance.hamming(np.array(a).flatten(), np.array(b).flatten())


def get_jensen_shannon_distance(a, b):
    distance.jensenshannon(np.array(a).flatten(), np.array(b).flatten())