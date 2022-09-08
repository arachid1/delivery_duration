import argparse
import os
import tensorflow as tf
import random
import numpy as np
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def seed():
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)


def initialize_file_folder(file_dir):
    print("File dir is {}".format(file_dir))
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
    os.mkdir(file_dir)


def init(arguments, file_name):

    print("-- Collecting Variables... --")
    print("Tensorflow Version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    global cache_root
    global data_root
    # cache_root = "../../doordash/prediction/cache/"
    cache_root = "/home/alirachidi/doordash/prediction/cache"
    data_root = "/home/alirachidi/doordash/data"
    # data_root = "../../doordash/data/"

    global job_id
    global file_dir
    global description
    global debugging
    job_id = 0
    file_dir = os.path.join(
        cache_root, file_name)
    description = arguments["description"]
    debugging = int(arguments["debugging"])

    global n_labels
    global initial_channels
    n_labels = 1
    initial_channels = 1

    global stack_length
    # global data_length
    stack_length = 1
    # data_length = 7

    global n_epochs
    global batch_size
    global es_patience
    global lr
    global lr_patience
    # global min_update # not used yet
    global factor
    global min_lr
    n_epochs = 200
    batch_size = 512
    es_patience = 6
    lr = 1e-3
    lr_patience = 3
    # min_update = 0.0
    factor = 0.5
    min_lr = 1e-5

    global ll2_reg
    global weight_decay
    ll2_reg = 1e-4
    weight_decay = 0

    global kfold
    global moving_window
    global normalize
    global reduce_size
    global use_weights
    kfold = True
    moving_window = True
    global n_folds
    global train_test_ratio
    train_test_ratio = 0.8
    n_folds = 5
    normalize = False
    reduce_size = False
    use_weights = False

    global feature_writing
    global stacking_writing
    global timer_feature
    global stacking_target
    global feature_version
    feature_writing = False  # define when to generate and write domain features
    stacking_writing = False  # defines when to write stacked samples
    timer_feature = False
    stacking_target = 'market_id'
    feature_version = "_v1"  # version of file to read/write to

    global task
    task = 1

    global n_minutes_treshold
    global long_threshold
    global short_threshold
    global loss_weight
    n_minutes_treshold = 10
    long_threshold = 3593  # 60 minutes, or 80th percentile / 3202.0   70th percentile
    short_threshold = 1987  # 33 minutes, or 20th percentil percentile / 2218.0   30th percentile
    loss_weight = 1.0

    global domain_columns
    global domain_settings
    domain_columns = ['market_id', 'store_id']
    domain_settings = []  # defined inside every main file (to avoid importing other files here)

    global decision_features
    global time_features
    global domain_features
    global same_space_features
    # default features
    decision_features = ['market_id', 'store_id', 'subtotal',
                         'total_onshift_dashers', 'total_busy_dashers',
                         'total_outstanding_orders',
                         'estimated_store_to_consumer_driving_duration']
    # an easy way to aggregate and manipulate features for selection at choice
    time_features = ['week_day', 'weekend', 'day_of_month', 'month', 'hour', 'day_part']
    domain_features = [
        '1_hr_count_by_market_id',
        '1_hr_count_by_store_id',
        '1_hr_avg_subtotal_by_market_id',
        '1_hr_avg_subtotal_by_store_id',
        '24_hr_avg_time_by_market_id',
        '24_hr_avg_time_by_store_id',
        'long_score_by_market_id',
        'short_score_by_market_id',
        'long_score_by_store_id',
        'short_score_by_store_id',
        'trip_to_customer_by_market_id',
        'trip_to_customer_by_store_id']
    # gives us flexibility about when to search in same space (i.e., 1_hr variables) or not (i.es, long_score, a historical variable)
    same_space_features = ['1_hr_count_by_market_id',
                           '1_hr_count_by_store_id',
                           '1_hr_avg_subtotal_by_market_id',
                           '1_hr_avg_subtotal_by_store_id',
                           '24_hr_avg_time_by_market_id',
                           '24_hr_avg_time_by_store_id']

    # layout for tensorboard
    global layout
    layout = {
        "Metrics": {
            "train_loss_batch": ["Multiline", ["batch_loss/train_batch"]],
            "train_loss": ["Multiline", ["loss/train"]],
            "val_loss": ["Multiline", ["loss/validation"]],
            "mean_absolute_error": ["Multiline", ["mean_absolute_error/train", "mean_absolute_error/validation"]],
            "lateness": ["Multiline", ["lateness/train", "lateness/validation"]],
            "earliness": ["Multiline", ["earliness/train", "earliness/validation"]],
        },
    }

    # defines when to run val loop while training or writing test results to a file after training
    global prediction_mode
    prediction_mode = False

    if debugging:
        file_dir += "_debugging"
        n_epochs = 4
        es_patience = 3
        train_test_ratio = 0.5
        lr_patience = 1
        description = "debugging"

    print("PID: {}".format(os.getpid()))

    print("Description: {}".format(description))
    initialize_file_folder(file_dir)
    print("--- All variables have been collected. --")


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debugging",
        help="activate debugging mode or not for debugging purposes",
        required=False
    )
    parser.add_argument(
        "--description",
        help="description of the job",
        required=False,
    )
    args = parser.parse_args()
    arguments = args.__dict__
    return arguments
