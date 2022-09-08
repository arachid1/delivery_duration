from collections import defaultdict
from modules.main import parameters as p
from modules.main.training import *
from modules.main.processing import *
from modules.main.feature_engineering import *
from modules.main.analysis import *
from modules.main.helpers import *
from modules.main.stacking import *
from modules.main.metrics import *
from modules.models.RNN2 import General_CNN_3


from sklearn.model_selection import RepeatedKFold
import tensorflow as tf
import os
import warnings
import sys
from io import StringIO

warnings.filterwarnings(
    "ignore",
    category=np.VisibleDeprecationWarning)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)

# TODO: comment
# TODO: remove import
# TODO: test function
# TODO: add assert statements
# TODO: change file names back from bis
# TODO: pass model that reiterate


def train_model(model_to_be_trained=None):

    historical_data = extract_samples(
        os.path.join(p.data_root, "historical_data.csv"),
        ['created_at', 'actual_delivery_time'])

    # TODO: add date features, how much time/if it ended, s/d score, avg delivery time, holidays
    historical_data = add_features(historical_data)

    X_hist = historical_data[p.decision_features]
    y_hist = historical_data['delivery_duration_sec'].astype('int32')

    X_hist, y_hist = stack_samples(historical_data, p.stacking_target)
    # Note: at this stage, input is split by p.stacking_target i.e. by market or store

    kfold_metrics = defaultdict(lambda: [])

    kf = RepeatedKFold(n_splits=p.n_folds, n_repeats=1,
                       random_state=12)
    if not p.moving_window:
        # random kfold (destined for unserialized data with stack_length = 1) instead of slicing window
        X_hist = np.concatenate(X_hist, axis=0)
        y_hist = np.concatenate(y_hist, axis=0)
        splits = kf.split(X_hist, y_hist)

    #  TODO: handle singular random and sequential kfold
    for i in range(p.n_folds):

        initialize_job()

        if p.moving_window:
            # returns aggregate of ith slice from each p.stacking_target
            X_train, X_val, y_train, y_val = return_slice(X_hist, y_hist, i)
        else:
            # returns selections from all samples using kfold
            t, v = next(splits)
            X_train, X_val = X_hist[t], X_hist[v]
            y_train, y_val = y_hist[t], y_hist[v]
            # TODO: needs to be float32

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=p.lr)

        loss_fn = mean_squared_error

        metrics = [tf.keras.metrics.MeanAbsoluteError(), lateness(), earliness()]

        best_model, best_metrics = job_function(model_to_be_trained, X_train, y_train,
                                                X_val, y_val, loss_fn, optimizer, metrics)

        for (k, v) in best_metrics.items():
            kfold_metrics[k].append(v)

        if (not p.kfold):
            break

    print("-------")
    print("Kfold metrics")
    for (k, v) in kfold_metrics.items():
        avg = np.mean(v)
        std = np.std(v)
        max_val = max(v)
        min_val = min(v)
        print("{} mean: {}".format(k, avg))
        print("{} std: {}".format(k, std))
        print("{} max: {}".format(k, max_val))
        print("{} min: {}".format(k, min_val))


if __name__ == "__main__":

    p.seed()
    p.init(
        p.parse_arguments(),
        os.path.basename(__file__).split('.')
        [0])

    p.decision_features += ['week_day', 'weekend', 'day_of_month', 'month', 'hour', 'day_part',
                            '1_hr_count_by_store_id', '1_hr_avg_subtotal_by_store_id',
                            '24_hr_avg_time_by_store_id', '1_hr_count_by_market_id',
                            '1_hr_avg_subtotal_by_market_id', '24_hr_avg_time_by_market_id',
                            #'trip_to_customer_by_market_id', 'trip_to_customer_by_store_id'
                            ]

    p.feature_writing = False
    p.feature_v = "_v2"

    p.moving_window = True

    p.stacking = True
    p.stack_length = 10
    p.stacking_target = "market_id"

    p.timer_feature = True

    train_model(General_CNN_3)
