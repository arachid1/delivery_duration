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

    parameters.features = ['market_id', 'store_id', "subtotal",
                           "total_onshift_dashers", "total_busy_dashers",
                           "total_outstanding_orders",
                           "estimated_store_to_consumer_driving_duration"]

    if parameters.feature_augmentation:
        print("Augmenting...")
        historical_data, new_features = add_features(historical_data)
        parameters.features = parameters.features+new_features
        parameters.data_shape = (None, parameters.stack_length, len(parameters.features))
        print("{} features have been added: {}".format(len(new_features), new_features))

    historical_data = historical_data.sample(frac=1).reset_index(drop=True)
    X_hist, y_hist = prepare_samples(historical_data, parameters.features)
    # X_test, _ = prepare_samples(test_data, parameters.features[:parameters.data_length])

    historical_data = remove_outliers(historical_data)
    kf = RepeatedKFold(n_splits=5, n_repeats=1,
                       random_state=12)
    maes = []

    # print(historical_data.shape)
    # print(historical_data.sort_values(
    #     by=['store_id', 'created_at'], ascending=[True, True]).head(10))
    # print("inbtw")
    # print(historical_data.sort_values(
    #     by=['store_id', 'created_at'], ascending=[True, True]).tail(10))
    # print(historical_data['prev_orders_avg_by_store_id'].value_counts())
    # print(historical_data['prev_orders_count_by_market_id'].value_counts())
    # print(historical_data['prev_orders_count_by_store_id'].value_counts())
    # exit()

    if parameters.scale:
        print("Scaling...")
        scaling = StandardScaler().fit(X_hist)
        X_hist = scaling.transform(X_hist)

    for t, v in kf.split(X_hist, y_hist):

        X_train, X_val = X_hist[t], X_hist[v]
        y_train, y_val = y_hist[t], y_hist[v]

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

        model = model_to_be_trained()
        # model.fit(X_train, y_train)
        model.build(parameters.data_shape)
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
    parameters.domains = ['market_id', 'store_id']

    parameters.reduce_size = False
    parameters.scale = False
    parameters.kfold = False

    parameters.weight_decay = 0
    parameters.lr = 1e-3
    parameters.ll2_reg = 1e-4
    # parameters.factor = 0.1
    parameters.batch_size = 512

    parameters.sequential = tf.keras.Sequential(
        [tf.keras.layers.Dense(
            512, activation="sigmoid", activity_regularizer=tf.keras.regularizers.l2(
                parameters.ll2_reg)),
         tf.keras.layers.Dropout(0.1),
         tf.keras.layers.Dense(
             256, activation="sigmoid", activity_regularizer=tf.keras.regularizers.l2(
                 parameters.ll2_reg)),
         tf.keras.layers.Dropout(0.1),
         tf.keras.layers.Dense(
             128, activation="sigmoid", activity_regularizer=tf.keras.regularizers.l2(
                 parameters.ll2_reg)),
         tf.keras.layers.Dropout(0.1),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(parameters.n_labels, activation="relu")])

    train_model(General_CNN)
    initialize_job()
