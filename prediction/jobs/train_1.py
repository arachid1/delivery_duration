from modules.main import parameters
from modules.main.parameters import initialize_job
from modules.main.training import *
from modules.main.processing import *
from modules.main.analysis import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import os
import warnings
import sys
warnings.filterwarnings(
    "ignore",
    category=np.VisibleDeprecationWarning)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


def train_model(model_to_be_trained):

    historical_data = extract_samples(
        os.path.join(parameters.data_root, "historical_data.csv"),
        ['created_at', 'actual_delivery_time'])
    test_data = extract_samples(
        os.path.join(parameters.data_root, "predict_data.csv"),
        ['created_at'])

    features = ['market_id', 'store_id', "subtotal", "total_onshift_dashers",
                "total_busy_dashers", "total_outstanding_orders",
                "estimated_store_to_consumer_driving_duration"]

    original_length = len(features)

    if parameters.augment_features:
        historical_data, new_features = add_features(historical_data)
        features = features+new_features

    X_hist, y_hist = prepare_samples(historical_data, features)
    X_test, _ = prepare_samples(test_data, features[:original_length])
    maes = []

    kf = RepeatedKFold(n_splits=5, n_repeats=1,
                       random_state=12)

    for t, v in kf.split(X_hist, y_hist):

        X_train, X_val = X_hist.iloc[t], X_hist.iloc[v]
        y_train, y_val = y_hist.iloc[t], y_hist.iloc[v]

        model = model_to_be_trained()
        model.fit(X_train, y_train)
        regr_coeffs = model.coef_

        res = "\n".join("{} {}".format(x, y) for x, y in zip(features, regr_coeffs))

        plot_regression_coeffs(regr_coeffs, features)

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)

        diff = y_val-y_pred
        diff.drop(index=diff.nlargest(5).index, inplace=True)

        plot_box_plot(abs(diff), _id="diff_box")

        plot_bar_plot(diff, _id="diff_bar", title="y_val-ypred", lims=(-2000, 3000),
                      y_label="difference between pre diction and label")

        print("Val results")
        print(res)
        print('mean absolute error %f' % mae)

        maes.append(mae)

        if (not parameters.kfold):
            break
        initialize_job()

    print("Kfold metrics")
    kfold_mae = sum(maes) / len(maes)
    print("Kfold average mae: %f " % kfold_mae)


if __name__ == "__main__":

    parameters.seed()
    parameters.init(
        parameters.parse_arguments(),
        os.path.basename(__file__).split('.')
        [0])

    parameters.augment_features = True
    parameters.generate_last_orders = False

    train_model(
        LinearRegression)
