import tensorflow as tf
from keras import layers
from keras.regularizers import l2
from ..main import parameters as p
import os


class RNN7(tf.keras.Model):

    def __init__(self):

        super().__init__()
        self._model = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(
                8, kernel_size=(3, 3), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(
                16, kernel_size=(3, 3), padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(
                64, kernel_size=(3, 3), padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Reshape((p.stack_length, -1)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                # tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(256, activation="relu",  activity_regularizer=l2(
                    p.ll2_reg), bias_regularizer=l2(
                    p.ll2_reg)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                p.n_labels)]
        )

    def call(self, inputs, training=None):

        output = inputs
        # print(output.shape)
        output = tf.expand_dims(output, -1)
        # output = tf.squeeze(output, axis=1)
        # print(output.shape)
        output = tf.repeat(output, repeats=[p.initial_channels], axis=-1)
        # print(output.shape)
        output = self._model(output)
        return output

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # print(y_pred)
            sample_weight = self.compute_weights(y)
            loss_value = self.compiled_loss(y, y_pred, sample_weight=sample_weight)
            # loss_value += sum(self.losses)
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
        if p.use_weights:
            sample_weight = tf.map_fn(fn=lambda t: tf.gather(
                p.weights, tf.cast(t, tf.int32)), dtype=tf.float64, elems=y)
        return sample_weight

    def save(self, dest, epoch):
        self._model.save_weights(os.path.join(dest, "_model_{}.h5".format(epoch)))

    def _load(self, source, epoch):
        self._model.load_weights(os.path.join(source, "_model_{}.h5".format(epoch)))
