import tensorflow as tf
from . import parameters as p
import keras.backend as K

'''
custom mean squared error that can applie a coefficient to long or short orders
:param y_true: true labels (tf.Tensor)
:param y_pred: predicted labels (tf.Tensor)
'''


def mean_squared_error(y_true, y_pred):
    sample_weight = tf.map_fn(fn=lambda t: (tf.math.less_equal(
        t, p.short_threshold) or tf.math.greater(
        t, p.long_threshold)), dtype=tf.bool, elems=y_pred)
    sample_weight = tf.where(sample_weight, p.loss_weight, 1)
    mse = K.square(y_pred - y_true)
    mse = tf.multiply(mse, sample_weight)
    return K.mean(mse, axis=-1)
