from collections import defaultdict
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
from io import StringIO

warnings.filterwarnings(
    "ignore",
    category=np.VisibleDeprecationWarning)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 500)

# TODO: clean
# TODO: comment
# TODO: remove import
# TODO: test function


def train_model(model_to_be_trained=None):

    historical_data = extract_samples(
        os.path.join(p.data_root, "_historical_data.csv"),
        ['created_at', 'actual_delivery_time'])

    historical_data = add_features(historical_data)

    # historical_data.sort_values(
    #     by=['store_id', 'created_at'],
    #     ascending=[True, True],
    #     inplace=True)
    # historical_data = historical_data.tail(10)
    # # historical_data.head().__repr__().to_csv("test.txt", header=True, index=False, sep="\t")
    # with open('./filename.txt', 'w') as fo:
    #     fo.write(historical_data.__repr__())
    # exit()
    X_hist = historical_data[p.decision_features]
    y_hist = historical_data['delivery_duration_sec'].astype('int32')

    # stack_samples returns a np.array of np.arrays for each unique domain e.g. market id or store id
    X_hist, y_hist = stack_samples(historical_data, p.stacking_target)

    kfold_metrics = defaultdict(lambda: [])

    if not p.moving_window:
        # random kfold (destined for unserialized data with stack_length = 1) instead of slicing window scheme
        kf = RepeatedKFold(n_splits=p.n_folds, n_repeats=1,
                           random_state=12)
        X_hist = np.concatenate(X_hist, axis=0)
        y_hist = np.concatenate(y_hist, axis=0)
        splits = kf.split(X_hist, y_hist)

    for i in range(p.n_folds):

        initialize_job()

        if p.moving_window:
            # returns respective ith slice from each p.stacking_target i.e. 1st slice of each store
            X_train, X_val, y_train, y_val = return_slice(X_hist, y_hist, i)
        else:
            # iteratively generates random unique kfolds
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

    p.domain_settings = [("{}_hr_count_by_".format(1), return_count, 1),
                         ("{}_hr_avg_subtotal_by_".format(1), return_avg_subtotal, 1),
                         ("{}_hr_avg_time_by_".format(24), return_avg_time, 24),
                         ("long_score_by_", return_long_score, 730),
                         ("short_score_by_", return_short_score, 730),
                         ("trip_to_customer_by_", return_trip_to_customer, 730)]

    # p.decision_features += p.time_features
    # p.decision_features += p.domain_features

    p.feature_writing = False
    p.feature_version = "_v4"
    p.loss_weight = 1.0
    p.task = 1

    p.moving_window = True

    p.stacking = True
    p.stack_length = 3
    p.stacking_target = "market_id"

    p.batch_size = 512
    p.kfold = False

    # p.timer_feature = True

    # first is a time feature of most recent and sensitive information about market order
    # second is larger scale trends across days, such as weather decline or sudden increase in orders
    # third is regresion model prediction of the trip from dasher to customer based on trip from dasher to store
    # fourth is 0-1 score for historical amount of long or short prediction, decided with threshold of 70-90th percentiles
    features_blobs = [['1_hr_count_by_market_id'],
                      ['1_hr_count_by_store_id'],
                      ['1_hr_avg_subtotal_by_market_id'],
                      ['1_hr_avg_subtotal_by_store_id'],
                      ['24_hr_avg_time_by_market_id'],
                      ['24_hr_avg_time_by_store_id'],
                      ['trip_to_customer_by_market_id'],
                      ['trip_to_customer_by_store_id'],
                      ['long_score_by_market_id'],
                      ['short_score_by_market_id'],
                      ['long_score_by_store_id'],
                      ['short_score_by_store_id']]

    for t in [True, False]:
        for blob in features_blobs:
            p.decision_features = ['market_id', 'store_id', 'subtotal',
                                   'total_onshift_dashers', 'total_busy_dashers',
                                   'total_outstanding_orders',
                                   'estimated_store_to_consumer_driving_duration']
            p.decision_features += p.time_features
            p.decision_features += blob
            p.timer_feature = t
            print("info")
            print(p.decision_features)
            print(len(p.decision_features))
            print(p.timer_feature)
            train_model(RNN8)
