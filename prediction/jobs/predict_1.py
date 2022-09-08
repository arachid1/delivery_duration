from collections import defaultdict
from re import A
from modules.main import parameters as p
from modules.main.training import *
from modules.main.processing import *
from modules.main.feature_engineering import *
from modules.main.analysis import *
from modules.main.helpers import *
from modules.main.stacking import *
from modules.main.metrics import *
from modules.main.losses import *
from modules.models.RNN8 import RNN8


from sklearn.model_selection import RepeatedKFold
import tensorflow as tf
import os
import warnings
import sys

warnings.filterwarnings(
    "ignore",
    category=np.VisibleDeprecationWarning)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)


def train_model(model_to_be_trained=None):

    historical_data = extract_samples(
        os.path.join(p.data_root, "_historical_data.csv"),
        ['created_at', 'actual_delivery_time'])
    test_data = extract_samples(
        os.path.join(p.data_root, "_predict_data.csv"),
        ['created_at'])

    historical_data = add_features(historical_data)
    p.feature_writing = True
    copy = historical_data
    copy['delivery_id'] = 1  # adding dimensions to search space for compatibility
    test_data = add_features(test_data, label="test_", search_spaces={
                             'historical': copy})
    # write_short_sample(test_data.head(), "test_file.txt")

    # not a necessary step because stack samples uses p.decision features to extract final features
    X_hist = historical_data[p.decision_features]
    y_hist = historical_data['delivery_duration_sec'].astype('int32')
    X_test = test_data[p.decision_features]

    # stack_samples returns a np.array of np.arrays for each unique domain e.g. market id or store id
    X_hist, y_hist = stack_samples(historical_data, p.stacking_target)
    X_test, _ = stack_samples(test_data, p.stacking_target)
    X_hist = np.array([x for batch in X_hist for x in batch]).astype('float32')
    y_hist = np.array([y for batch in y_hist for y in batch]).astype('float32')
    y_hist = np.expand_dims(y_hist, axis=-1)
    X_test = np.array([x for batch in X_test for x in batch]).astype('float32')

    initialize_job()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=p.lr)

    loss_fn = mean_squared_error

    metrics = [tf.keras.metrics.MeanAbsoluteError(), lateness(), earliness()]

    _, _ = job_function(model_to_be_trained, X_hist, y_hist,
                        np.array([]), np.array([]), loss_fn, optimizer, metrics)


if __name__ == "__main__":

    p.seed()
    p.init(
        p.parse_arguments(),
        os.path.basename(__file__).split('.')
        [0])

    p.decision_features += p.time_features
    p.decision_features += p.domain_features
    p.decision_features.remove('24_hr_avg_time_by_market_id')
    p.decision_features.remove(
        '24_hr_avg_time_by_store_id')

    p.domain_settings = [
        ("{}_hr_count_by_".format(1),
         return_count, 1, 'add', ['historical']),
        ("{}_hr_avg_subtotal_by_".format(1),
         return_avg_subtotal, 1, 'add', ['historical']),
        ("long_score_by_", return_long_score, 730, 'historical', None),
        ("short_score_by_", return_short_score, 730, 'historical', None),
        ("trip_to_customer_by_", return_trip_to_customer, 730, 'historical', None)]

    p.feature_writing = False
    p.feature_version = "_v4"
    p.loss_weight = 1.0
    p.task = 1

    p.moving_window = True

    p.stacking = True
    p.stack_length = 3
    p.stacking_target = "market_id"

    p.batch_size = 512

    p.timer_feature = False
    p.prediction_mode = True
    train_model(RNN8)
