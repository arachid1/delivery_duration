from pyexpat import features
from modules.main import parameters
from modules.main.parameters import initialize_job
from modules.main.training import *
from modules.main.processing import *
from modules.main.analysis import *
from modules.models.General_CNN import General_CNN
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
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


def train_model(model_to_be_trained=None):

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

    # parameters.features = ['market_id', 'created_at', 'store_id', "subtotal",
    #                        "total_onshift_dashers", "total_busy_dashers",
    #                        "total_outstanding_orders",
    #                        "estimated_store_to_consumer_driving_duration"]

    if parameters.feature_augmentation:
        print("Augmenting...")
        historical_data, new_features = add_features(historical_data)
        parameters.features = parameters.features+new_features
        # parameters.data_shape = (None, parameters.stack_length, len(parameters.features))
        print("{} features have been added: {}".format(len(new_features), new_features))

    historical_data = historical_data.sample(frac=1).reset_index(drop=True)  # shuffle

    historical_data = remove_outliers(historical_data)  # removing postprocessing for flexibility

    # X_hist, y_hist = prepare_samples(historical_data, parameters.features)
    # X_test, _ = prepare_samples(test_data, parameters.features[:parameters.data_length])

    if parameters.stacking:
        X_hist, y_hist = stack_samples(historical_data, parameters.stacking_target)

    # X_hist = X_hist.to_numpy()
    # y_hist = y_hist.to_numpy()

    kf = RepeatedKFold(n_splits=5, n_repeats=1,
                       random_state=12)
    maes = []

    if parameters.scale:
        print("Scaling...")
        scaling = StandardScaler().fit(X_hist)
        X_hist = scaling.transform(X_hist)

    for i in range(parameters.n_folds):
        X_train = []
        X_val = []
        y_train = []
        y_val = []
        for y in range(len(X_hist)):
            domain_wise_X = X_hist[y]
            domain_wise_y = y_hist[y]
            # print(len(domain_wise))
            window_shift = int(len(domain_wise_X)/parameters.n_folds)
            lower = i*window_shift
            upper = (i+1)*window_shift
            val_indexes = [v for v in range(lower, upper)]
            train_indexes = list(
                range(0, lower)) + list(range(lower+window_shift, len(domain_wise_X)))
            # print(train_indexes)
            # print(val_indexes)
            X_d_train, X_d_val = domain_wise_X[train_indexes], domain_wise_X[val_indexes]
            y_d_train, y_d_val = domain_wise_y[train_indexes], domain_wise_y[val_indexes]
            X_train.append(X_d_train)
            X_val.append(X_d_val)
            y_train.append(y_d_train)
            # print(y_d_txsrain.shape)
            y_val.append(y_d_val)
        X_train = np.concatenate(X_train)
        X_val = np.concatenate(X_val)
        y_train = np.concatenate(y_train)
        y_val = np.concatenate(y_val)
        # print(y_train.shape)
        # exit()

        if parameters.reduce_size:
            train_length = len(X_train)
            X_train = X_train[:int(train_length*0.01)]
            y_train = y_train[:int(train_length*0.01)]

        train_dataset = create_tf_dataset(
            X_train, y_train, shuffle=True)
        val_dataset = create_tf_dataset(
            X_val, y_val, shuffle=False)

        print_dataset(y_train, y_val)

        # weights
        if parameters.use_weights:
            compute_class_weights(y_train)

        # padding = "same"
        # pool_size = (1, 1)
        # kernel_size = (2, 15)
        # activation = "relu"
        parameters.sequential = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(
                32, kernel_size=(3, 1), padding="same", activation="relu"),
             tf.keras.layers.Conv2D(
                64, kernel_size=(3, 1), padding="same", activation="relu"),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Dropout(0.1),
             #  tf.keras.layers.Conv1D(
             #      16, kernel_size=kernel_size, padding=padding, activation=activation),
             #  tf.keras.layers.Conv1D(
             #      32, kernel_size=kernel_size, padding=padding, activation=activation),
             #  tf.keras.layers.MaxPooling2D(pool_size=pool_size),
             #  tf.keras.layers.BatchNormalization(),
             #  tf.keras.layers.Dropout(0.1),
             tf.keras.layers.Reshape((parameters.stack_length, -1)),
             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
             tf.keras.layers.Dropout(0.1),
             tf.keras.layers.Dense(512, activation="relu"),
             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(
                 parameters.n_labels, activation="relu")])

        model = model_to_be_trained()
        # model.fit(X_train, y_train)
        model.build((None, parameters.stack_length, len(parameters.features)))
        model._model.summary()

        # optimizer = tf.keras.optimizers.SGD(
        #     learning_rate=parameters.lr, decay=parameters.weight_decay, momentum=0.9,
        #     nesterov=False, name="SGD")

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=parameters.lr)

        loss_fn = tf.keras.losses.MeanAbsoluteError()

        metrics = [tf.keras.metrics.MeanAbsoluteError()]

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

        best_model, mae = train_function(model, loss_fn, optimizer, train_dataset,
                                         val_dataset, writer)

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

    parameters.feature_augmentation = True
    parameters.feature_generation = False
    parameters.domains = ['market_id', 'store_id']

    parameters.reduce_size = False
    parameters.scale = False
    parameters.kfold = True
    parameters.n_folds = 5

    parameters.weight_decay = 0
    parameters.lr = 1e-3
    parameters.ll2_reg = 1e-4
    parameters.batch_size = 512

    parameters.stacking = True
    parameters.stack_length = 3
    parameters.stacking_target = "market_id"

    train_model(General_CNN)
    initialize_job()
