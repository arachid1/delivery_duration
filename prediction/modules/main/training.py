from . import parameters as p
import tensorflow as tf
import time
import math


def display_metrics(writer, model, loss_avg, epoch, mode="train"):
    '''
    computes, prints and logs metrics and losses for both training and validation
    it also resets theit states
    :param writer: tensorboard writer used for the given (torch.utils.tensorboard.SummaryWriter)
    :param model: model (tf.keras.Model)
    :param loss_avg: batch loss average (tf.Tensor)
    :param epoch: epoch number (int)
    :param mode: indicates train or val for printing and logs (str)
    :return: metrics (dict) containing each name (key) and correspond value

    '''
    writer.add_scalar('loss/{}'.format(mode), loss_avg.numpy(), epoch)
    tf.print("{} loss: {}".format(mode, loss_avg))
    metrics = {}
    for m in model.compiled_metrics._metrics:
        value = m.result().numpy()
        tf.print("{} {}: {}".format(mode, m.name, value))
        metrics[m.name] = value
        writer.add_scalar("{}/{}".format(m.name, mode), value, epoch)
        m.reset_state()
    return metrics


def update_training(wait, tracker, best, optimizer):
    '''
    get calles when better results haven't been achieved from last epoch, so therefore
    it updates training configuration such as learning rate but also stops the job if performance doesn't improve for long
    :param wait: number of epochs waited without an improvement (int)
    :param tracker: tracker for number of epochs without improvement at current lrs (int)
    :param best: best value of target metric (float)
    :param optimizer: optimizer  (tf.keras.optimizers.Optimizer)
    :return: stop_training boolean to halt or continue the training
    '''
    if wait >= p.es_patience:
        print("Training stopped due to unimproved results over {} epochs".format(
            p.es_patience))
        return 1
    else:
        print("The validation tracker metric at {} hasn't increased  in {} epochs".format(
            best, tracker))
        if (not (tracker == 0)) and tracker % p.lr_patience == 0:
            if optimizer.lr > p.min_lr:
                optimizer.lr = optimizer.lr * p.factor
                print("Lr has been adjusted to {}".format(
                    optimizer.lr.numpy()))
        return 0


def train_function(model, optimizer, train_dataset, val_dataset, writer):
    '''
    low-level tensorflow training loop for effiency and flexibility on its principal tasks, the training and validation loop
    handles gradient backpropagation, loss and metric computation, and metrics logging to Tensorboard in conjuction with
    tf.keras.Model tools in the model passed (such as its own train_step and test_step function)
    supports adaptive learning rate and early stopping; saves best model and metrics (decided on a best target metric)
    :param model: built model to train (tf.keras.Model)
    :param optimizer: optimizer  (tf.keras.optimizers.Optimizer)
    :param train_dataset: train dataset object (tf.data.Dataset)
    :param val_dataset: val dataset object (tf.data.Dataset)
    :param writer: tensorboard writer used for the given (torch.utils.tensorboard.SummaryWriter)
    :return: best_model from best epoch (tf.keras.Model)
    :return: best metrics from best epoch (dict)
    '''
    train_length = len(train_dataset)

    best_model = None
    best_metrics = {}
    wait = 0
    tracker = 0
    best_epoch = 0
    best = math.inf

    for epoch in range(p.n_epochs):

        print("\nEpoch {}/{}".format(epoch, p.n_epochs))
        start_time = time.time()
        train_loss = 0.0
        val_loss = 0.0

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = model.train_step(x_batch_train, y_batch_train)
            # if step % 2000 == 0:
            #     print("loss at step {}: {}".format(step, loss_value))
            writer.add_scalar('batch_loss/train_batch', loss_value.numpy(),
                              epoch*train_length+step)
            train_loss += loss_value
        train_loss_avg = train_loss / (step+1)

        train_metrics = display_metrics(writer, model, train_loss_avg, epoch)

        print("Time taken for training epoch: %.2fs" % (time.time() - start_time))

        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            loss_value = model.test_step(x_batch_val, y_batch_val)
            val_loss += loss_value
        val_loss_avg = val_loss / (step+1)

        target_metrics = train_metrics

        if not p.prediction_mode:
            val_metrics = display_metrics(writer, model, val_loss_avg, epoch, "validation")
            target_metrics = val_metrics

        wait += 1
        if target_metrics['mean_absolute_error']-best < 0:
            best = target_metrics['mean_absolute_error']
            best_metrics = target_metrics
            best_model = model
            best_epoch = epoch
            wait = 0
            tracker = 0
            continue
        tracker += 1
        stop_training = update_training(wait, tracker, best, optimizer)
        if stop_training:
            break

    print("--- Best performance found at epoch {} --".format(best_epoch))
    print("Best value of tracked metric: {}".format(best))
    for k, v in best_metrics.items():
        print("Best {}: {}".format(k, v))

    if best_model is not None or p.debugging:
        best_model.save(p.job_dir, best_epoch)
        print("Best model saved...")

    return best_model, best_metrics
