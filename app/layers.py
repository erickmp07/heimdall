import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance layer from face_id module
# It's need to load the custom model
class L1Dist(Layer):
    """
    Siamese L1 Distance class
    """

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        """
        Similarity calculation
        """
        return tf.math.abs(input_embedding - validation_embedding)