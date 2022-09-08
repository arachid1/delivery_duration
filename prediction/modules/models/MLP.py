import tensorflow as tf
from .core import *
from keras import layers
from keras.regularizers import l2
from ..main import parameters
import os


class MLP(tf.keras.Model):

    def __init__(self):

        super().__init__()
        self._model = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(parameters.n_labels, activity_regularizer=l2(
                parameters.ll2_reg), activation="sigmoid")
        ])

    def call(self, inputs, training=None):

        output = inputs
        if parameters.normalize:
            output = output - tf.math.reduce_min(output)
            output = output/tf.math.reduce_max(output)
        output = self._model(output)
        output = tf.reshape(output, (-1, parameters.n_labels))
        return output

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            sample_weight = self.compute_weights(y)
            loss_value = self.compiled_loss(y, y_pred, sample_weight=sample_weight)
            loss_value += sum(self.losses)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for m in self.compiled_metrics._metrics:
            m.update_state(y, y_pred)
        return loss_value

    def test_step(self, x, y):
        y_pred = self(x, training=False)
        sample_weight = self.compute_weights(y)
        loss_value = self.compiled_loss(y, y_pred, sample_weight=sample_weight)
        loss_value += sum(self.losses)
        for m in self.compiled_metrics._metrics:
            m.update_state(y, y_pred)
        return loss_value

    def compute_weights(self, y):
        sample_weight = None
        if parameters.use_weights:
            sample_weight = tf.map_fn(fn=lambda t: tf.gather(
                parameters.weights, tf.cast(t, tf.int32)), dtype=tf.float64, elems=y)
        return sample_weight

    def save(self, dest, epoch):
        self._model.save_weights(os.path.join(dest, "_model_{}.h5".format(epoch)))

    def _load(self, source, epoch):
        self._model.load_weights(os.path.join(source, "_model_{}.h5".format(epoch)))
