import numpy as np
from . import parameters as p
from .training import *
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.preprocessing import StandardScaler


def return_slice(X_hist, y_hist, i):
    '''
    returns (i+1)th slice of each np.array in X_hist corresponding to a distinct domain.
    In a stacking job (task 1b or 2), it is imperative to split data in distinct consecutive time periods,
    to train on some and validate on others.
    X_hist is assumed to be given in chronological order so the slice is returned accordingly.

    :param X_hist: stacked samples separated by domains (np.array of np.arrays)
    :param y_hist: corresping labels (np.array of np.arrays)
    :param i: ith slice generated from total p.n_folds slices (np.array of np.arrays)
    :returns: X_train (np.array), X_val (np.array), y_train (np.array), y_val (np.array)
    '''
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    for y in range(len(X_hist)):  # iterating to split each individual domain separately
        domain_X = X_hist[y]
        domain_y = y_hist[y]
        w = int(len(domain_X)/p.n_folds)  # window shift
        lower = i*w
        upper = (i+1)*w
        v = [v for v in range(lower, upper)]  # val indexes
        t = list(
            range(0, lower)) + list(range(lower+w, len(domain_X)))  # train indexes
        X_train.append(domain_X[t])
        X_val.append(domain_X[v])
        y_train.append(domain_y[t])
        y_val.append(domain_y[v])

    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    y_train = np.concatenate(y_train)
    y_val = np.concatenate(y_val)
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    return np.float32(X_train), np.float32(X_val), np.float32(y_train), np.float32(y_val)


def job_function(
        model_to_be_trained, X_train, y_train, X_val, y_val, loss_fn, optimizer, metrics):
    '''
    this function handles the entire job by initializing required job parameters, instantiating model,
    perfoming operations such as reducing data, and calling the custom training loop function
    :param model_to_be_trained: model (tf.keras.Model)
    :param X_train: train input values (np.array)
    :param y_train: train lbale values (np.array)
    :param X_val: validation input values (np.array)
    :param y_val: validation lbale values (np.array)
    :loss_fn: loss object (tf.keras.losses.Loss or function object)
    :optimizer: optimizer  (tf.keras.optimizers.Optimizer)
    :metrics: list of function or keras metric objects to evaluate job on (list)
    :return: best_model from best epoch (tf.keras.Model)
    :return: best metrics from best epoch (dict)
    '''

    writer = SummaryWriter(
        os.path.join(p.job_dir, "logs"))
    writer.add_custom_scalars(p.layout)

    if p.normalize:
        print("Normalizing...")
        print(X_hist)
        scaling = StandardScaler().fit(X_hist)
        X_hist = scaling.transform(X_hist)

    if p.reduce_size:
        train_length = len(X_train)
        X_train = X_train[:int(train_length*0.1)]
        y_train = y_train[:int(train_length*0.1)]

    train_dataset = create_tf_dataset(
        X_train, y_train, shuffle=True)
    val_dataset = create_tf_dataset(
        X_val, y_val, shuffle=False)

    print_dataset(X_train, X_val)

    print("Model: {}".format(model_to_be_trained))
    print("Loss: {}\nloss weight: {}".format(loss_fn, p.loss_weight))
    model = model_to_be_trained()

    model.build((None, p.stack_length, X_train.shape[-1]))
    model._model.summary()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )

    best_model, best_metrics = train_function(model, optimizer, train_dataset,
                                              val_dataset, writer)

    writer.close()

    return best_model, best_metrics


def create_tf_dataset(X, y, shuffle=False, parse_func=None, prefetch_amount=8):
    '''
    this function converts inputs X and y to tf.data.Dataset object with additional parametrization,
    such as shuffling, parsing function or setting batch size
    :param X: inputs (np.array)
    :param y: labels (np.array)
    :param shuffle: determines if object should shuffle data (bool)
    :param parse_func: func to pre-process or parse the inputs/labels (function)
    :param prefetch_amount: numbers of batches to return (int)
    :return: a dataset ready for use by tf keras model (tf.data.Dataset)
    '''
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(len(y))
    if parse_func is not None:
        dataset = dataset.map(
            lambda sample, label: parse_func(sample, label),
            num_parallel_calls=16)
    dataset = dataset.batch(p.batch_size)
    dataset = dataset.prefetch(p.batch_size*prefetch_amount)
    return dataset


def print_dataset(X_train, X_val):
    '''
    custom print function for details on dataset and difference between splits
    :param X_train: train dataset (np.array)
    :param X_val: val dataset (np.array)
    '''
    print("--- training dataset has shape: {} ---".format(
        X_train.shape))
    print("--- validation dataset has shape: {} ---".format(X_val.shape))


def initialize_job():
    '''
    creates a job directory and updates parameters to synchronize for possible any additional job afterwards, and
    ensure multiple jobs can be ran in one file, whether in kfold or multi-experiment scheme
    '''
    p.job_dir = os.path.join(p.file_dir, str(p.job_id))
    print("-------------------------------")
    print("New job directory is {}".format(p.job_dir))
    os.mkdir(p.job_dir)
    os.mkdir(os.path.join(p.job_dir, "logs"))
    os.mkdir(os.path.join(p.job_dir, "others"))
    p.job_id += 1


def write_short_sample(df, file_name):
    '''
    writes a custom section of a given dataframe in a nice tabulated format for analysis and unittest use cases
    :param df: dataframe (pandas.DataFrame)
    '''
    df.sort_values(
        by=['market_id', 'created_at', 'store_id'],
        ascending=[True, True, True],
        inplace=True)
    df = df.head(10)
    with open(file_name, 'w') as fo:
        fo.write(df.__repr__())
