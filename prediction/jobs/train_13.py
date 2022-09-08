from modules.main import parameters
from modules.main.parameters import initialize_job
from modules.main.training import *
from modules.main.processing import *
from modules.main.analysis import *
from sklearn import svm
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import tensorflow as tf

import os
import warnings
import sys
warnings.filterwarnings(
    "ignore",
    category=np.VisibleDeprecationWarning)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


def train_model(model_to_be_trained):

    writer = SummaryWriter(
        os.path.join(parameters.job_dir, "logs"))
    writer.add_custom_scalars(parameters.layout)

    historical_data = extract_samples(
        os.path.join(parameters.data_root, "historical_data.csv"),
        ['created_at', 'actual_delivery_time'])
    test_data = extract_samples(
        os.path.join(parameters.data_root, "predict_data.csv"),
        ['created_at'])

    if parameters.testing:
        historical_data = historical_data[:100]

    parameters.features = ['market_id', 'store_id', "subtotal",
                           "total_onshift_dashers", "total_busy_dashers",
                           "total_outstanding_orders",
                           "estimated_store_to_consumer_driving_duration"]

    if parameters.augment_features:
        historical_data, new_features = add_features(historical_data)
        parameters.features = parameters.features+new_features
        parameters.data_shape = (None, parameters.stack_length, len(parameters.features))

    historical_data = historical_data.sample(frac=1).reset_index(drop=True)
    X_hist, y_hist = prepare_samples(historical_data, parameters.features)
    X_test, _ = prepare_samples(test_data, parameters.features[:parameters.data_length])

    kf = RepeatedKFold(n_splits=5, n_repeats=1,
                       random_state=12)
    maes = []

    for t, v in kf.split(X_hist, y_hist):

        X_train, X_val = X_hist.iloc[t], X_hist.iloc[v]
        y_train, y_val = y_hist.iloc[t], y_hist.iloc[v]

        train_length = len(X_train)
        X_train = X_train[:int(train_length*0.01)]
        y_train = y_train[:int(train_length*0.01)]

        print_dataset(y_train, y_val)

        # weights
        if parameters.use_weights:
            compute_class_weights(y_train)

        scaling = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
        print(X_train.iloc[0])
        X_train = scaling.transform(X_train)
        print(X_train[0])
        exit()
        X_val = scaling.transform(X_val)

        model = svm.SVC(kernel="linear")
        print("Fitting...")
        model.fit(X_train, y_train)
        print("Validating...")
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)

        diff = y_val-y_pred
        diff.drop(index=diff.nlargest(5).index, inplace=True)

        plot_box_plot(abs(diff), _id="diff_box")

        plot_bar_plot(diff, _id="diff_bar", title="y_val-ypred", lims=(-2000, 3000),
                      y_label="difference between pre diction and label")

        print("Val results")
        print('mean absolute error %f' % mae)

        maes.append(mae)

        writer.close()

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
        svm)
    initialize_job()
