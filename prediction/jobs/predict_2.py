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
    copy = historical_data
    copy['delivery_id'] = 1  # workaround but doesn't matter because only applied to search space
    p.feature_writing = True
    test_data = add_features(test_data, label="test_", search_spaces={
                             'historical': copy})

    # print(test_data[test_data['store_id'] == 39])
    # test_data.sort_values(by=['market_id', 'created_at'], ascending=[True, True], inplace=True)
    # # write_short_sample(test_data.head(), "test_file.txt")
    # for v in [1, 2, 3, 4, 5, 6]:
    #     for f in ['short_score_by_market_id']:
    #         print("new")
    #         print(v)
    #         temp = historical_data[historical_data['market_id'] == v]
    #         temp.sort_values(by=['market_id', 'created_at'], ascending=[True, True], inplace=True)
    #         d = test_data[test_data['market_id'] == v]
    #         d.sort_values(by=['market_id', 'created_at'], ascending=[True, True], inplace=True)
    #         d = d.reset_index()
    #         # print(temp.tail(1)[f])
    #         # print("test")
    #         # print(d.head(1)[f])
    #         # print(d.tail(1)[f])
    #         # print("mok")
    #         # print(d['market_id'].value_counts())
    #         print(d.head())
    #         # print(d[0]['created_at'])
    #         temp.to_csv("quick_check.csv")
    #         print(d.iloc[0]['created_at'])
    #         print(d.iloc[2000]['created_at'])
    #         print(d.iloc[5000]['created_at'])
    #         print(d.iloc[7500]['created_at'])
    #         print(d[f][:5000].value_counts())
    #         # print(d[0]['created_at'])
    #         print(d[f][5000:10000].value_counts())
    #         print(d[f][10000:15000].value_counts())
    #         print(d.tail())
    #         exit()

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

    best_model, _ = job_function(model_to_be_trained, X_hist, y_hist,
                                 np.array([]), np.array([]), loss_fn, optimizer, metrics)

    test_data.sort_values(
        by=[p.stacking_target, 'created_at'],
        ascending=[True, True],
        inplace=True)
    ids = test_data['delivery_id'].tolist()

    with open(os.path.join(p.job_dir, 'results.txt'), 'a') as f:
        for i, X in enumerate(X_test):
            # X = np.swapaxes(X, 0, 1)
            pred = best_model.predict([X])
            f.write('%d, %.2f\n' % (ids[i], pred))
            # print(ids[i])
            # print(X)
            # print(test_data[test_data['delivery_id'] == ids[i]])
            # if i == 3:
            #     exit()


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
        ("long_score_by_", return_long_score, 1460, 'historical', None),
        ("short_score_by_", return_short_score, 1460, 'historical', None),
        ("trip_to_customer_by_", return_trip_to_customer, 1460, 'historical', None)]

    p.feature_writing = False
    p.feature_version = "_v4"
    p.task = 1

    p.stack_length = 3

    p.timer_feature = False
    p.prediction_mode = True

    train_model(RNN8)
