import tensorflow as tf
from . import parameters as p

# both metrics alike but kept apart for now for flexibility

# metric that measures how often actual delivery occurs at least p.n_minutes_threshold minutes later than predicted delivery


class lateness(tf.keras.metrics.Metric):
    def __init__(self):
        super(lateness, self).__init__(name='lateness')
        self.n_minutes = p.n_minutes_treshold
        self.scores = []

    def reset_state(self):
        self.scores = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        diff = tf.math.subtract(y_true, y_pred)
        late = [e for e in diff if e > 60 * self.n_minutes]
        score = len(late) / len(diff)
        self.scores.append(score)

    def result(self):
        return tf.math.reduce_sum(self.scores) / tf.size(self.scores, out_type=tf.float32)


# metric that measures how often actual delivery occurs at least p.n_minutes_threshold minutes earlier than predicted delivery
class earliness(tf.keras.metrics.Metric):
    def __init__(self):
        super(earliness, self).__init__(name='earliness')
        self.n_minutes = p.n_minutes_treshold
        self.scores = []

    def reset_state(self):
        self.scores = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        diff = tf.math.subtract(y_true, y_pred)
        early = [e for e in diff if e < -60 * self.n_minutes]
        score = len(early) / len(diff)
        self.scores.append(score)

    def result(self):
        return tf.math.reduce_sum(self.scores) / tf.size(self.scores, out_type=tf.float32)
