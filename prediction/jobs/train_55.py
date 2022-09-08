from collections import defaultdict
from modules.main import parameters as p
from modules.main.training import *
from modules.main.processing import *
from modules.main.feature_engineering import *
from modules.main.analysis import *
from modules.main.helpers import *
from modules.main.stacking import *
from modules.main.metrics import *
from sklearn.linear_model import LinearRegression

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
    test_data = extract_samples(
        os.path.join(p.data_root, "predict_data.csv"),
        ['created_at'])

    # TODO: add date features, how much time/if it ended, s/d score, avg delivery time, holidays
    historical_data = add_features(historical_data)
    historical_data = remove_outliers(historical_data)  # removing postprocessing for flexibility
    # p.training_features = p.decision_features + ['avg_time_by_market_id',
    #                                              'avg_time_by_store_id']

    X_hist = historical_data[p.decision_features]
    y_hist = historical_data['delivery_duration_sec'].astype('float32')

    # test_data = add_features(test_data, "test_")
    # X_test = test_data[p.decision_features]

    print(X_hist.shape)

    # TODO: handle this or add to sep function
    # TODO: add variable when stacking
    X_hist, y_hist = stack_samples(historical_data, p.stacking_target)
    # X_test, _ = stack_samples(historical_data, p.stacking_target)
    # TODO: remove created_at

    kf = RepeatedKFold(n_splits=p.n_folds, n_repeats=1,
                       random_state=12)

    # kfold_metrics = defaultdict(lambda: [])
    maes = []

    #  TODO: handle singular random and sequential kfold
    for i in range(p.n_folds):

        initialize_job()

        X_train, X_val, y_train, y_val = return_slice(X_hist, y_hist, i)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=p.lr)

        loss_fn = tf.keras.losses.MeanSquaredError()

        metrics = [tf.keras.metrics.MeanAbsoluteError(), lateness(), earliness()]

        model = model_to_be_trained()
        X_train = np.squeeze(X_train)
        model.fit(X_train, y_train)
        regr_coeffs = model.coef_
        print("table")
        print(len(p.decision_features))
        print(p.decision_features)
        print(regr_coeffs)

        X_val = np.squeeze(X_val)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print("mae")
        print(mae)
        maes.append(mae)

        # best_model, best_metrics = job_function(model_to_be_trained, X_train, y_train,
        #                                         X_val, y_val, loss_fn, optimizer, metrics)

        # TODO: write test function if needed

        # for (k, v) in best_metrics.items():
        #     kfold_metrics[k].append(v)

        if (not p.kfold):
            break

    print("-------")
    print("Kfold metrics")
    print(sum(maes) / len(maes))
    # for (k, v) in kfold_metrics.items():
    #     avg = np.mean(v)
    #     std = np.std(v)
    #     max_val = max(v)
    #     min_val = min(v)
    #     print("{} mean: {}".format(k, avg))
    #     print("{} std: {}".format(k, std))
    #     print("{} max: {}".format(k, max_val))
    #     print("{} min: {}".format(k, min_val))


if __name__ == "__main__":

    p.seed()
    p.init(
        p.parse_arguments(),
        os.path.basename(__file__).split('.')
        [0])

    p.feature_writing = False
    # p.kfold = False

    p.stacking = False
    p.task = 1
    p.stack_length = 1
    p.stacking_target = "market_id"

    p.decision_features += ['week_day', 'hour',
                            'month', 'day_of_month',
                            'count_by_market_id',
                            'count_by_store_id',
                            'avg_subtotal_by_market_id',
                            'avg_subtotal_by_store_id',
                            'avg_time_by_market_id',
                            'avg_time_by_store_id'
                            ]

    # p.decision_features += ['week_day', 'weekend',
    #                         'day_of_month', 'month', 'hour', 'day_part',
    #                         'count_by_market_id',
    #                         'count_by_store_id',
    #                         'avg_subtotal_by_market_id',
    #                         'avg_subtotal_by_store_id',
    #                         'avg_time_by_market_id',
    #                         'avg_time_by_store_id'
    #                         ]

    train_model(LinearRegression)
