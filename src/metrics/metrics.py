import numpy as np
from scipy.spatial import distance
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

class CosineSimilarityLoss(Loss):
    def __init__(self, name='cosine_similarity_loss'):
        super(CosineSimilarityLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = K.l2_normalize(y_true, axis=-1)
        y_pred = K.l2_normalize(y_pred, axis=-1)
        return K.mean(1 - K.sum(y_true * y_pred, axis=-1))

# Define the custom MSE loss function
class MeanSquaredErrorLoss(Loss):
    def __init__(self, name='mean_squared_error_loss'):
        super(MeanSquaredErrorLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

def define_custom_loss():
    get_custom_objects().update({'cosine_similarity_loss': CosineSimilarityLoss})
    get_custom_objects().update({'mean_squared_error_loss': MeanSquaredErrorLoss})

def get_cosine_similarity(distribution_1, distribution_2):
    a = np.matrix.flatten(distribution_1)
    b = np.matrix.flatten(distribution_2)
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_hamming_distance(a, b):
    return distance.hamming(np.array(a).flatten(), np.array(b).flatten())


def get_jensen_shannon_distance(a, b):
    distance.jensenshannon(np.array(a).flatten(), np.array(b).flatten())