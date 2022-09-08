from . import parameters as p
import numpy as np
import time


def return_timer_feature(curr, counter, neighbor, timer_col, center=1):
    '''
    generates the (absolute) time difference between curr and its neighbor to indicate 'time distancing'
    between stacked samples. timer value the following constraints:
    - when stacking previous samples, 0 < timer < neighbor['delivery_duration_sec'] - 1 or timer = -1
    - e.g., if a previous order ended, timer set to -1 to indicate the order was completed
    :param curr: current order (pandas.dataSeries)
    :param counter: indicates where in the stack neighbor is located (int)
    :param neighbor: row in the stack compared to curr (pandas.dataSeries)
    :param timer_col: column of timer values to populate (np.array)
    :param center: user as a point reference to populate prev_post stacking both ways (int)
    :return: timer_col the column with updated timers (np.array)

    '''

    # time difference between the current element at i and its neighbor
    timer = abs((curr['created_at'] - neighbor['created_at']).total_seconds())
    # -1 means the order is completed and is assigned for previous elements in the stack only
    if timer >= neighbor['delivery_duration_sec'] and neighbor['created_at'] < curr['created_at']:
        timer = -1
    # counter is be negative for p.task == 1 (and can be for p.task == 2)
    if p.task == 1:
        timer_col[p.stack_length-1+counter] = timer
    else:
        timer_col[center+counter] = timer
    return timer_col


def stack_prev_and_post(s, i, domain_df, timer_col, center):
    '''
    for task 2, populates the stacked sample s by starting at the center of the stack, and moving in both directions
    to populate the stacked sample in one go with counter variable (from 1 to center).
    is built to add stack-dependent variables, such as timer_feature
    :param s: stacked_sample to populate (np.array)
    :param i: index of current element to help reference neighbors (int)
    :param domain_df: dataframe corresponding to current domain (pandas.dataFrame)
    :param timer_col: column of timer values to populate (np.array)
    :param center: user as a point reference to populate prev_post stacking both ways (int)
    :return: stacked sample (np.array)
    :return: timer_col, or the column of timer values to be concatenated (np.array)
    '''
    counter = 1
    temp_features, extra_features = add_temp_features(domain_df)
    curr = domain_df.iloc[i][temp_features]
    # populates both at the same time
    while counter <= center:
        # previous
        if i-counter >= 0:  # checking edges
            prev = domain_df.iloc[i-counter][temp_features]
            if p.timer_feature:
                timer_col = return_timer_feature(
                    curr, -counter, prev, timer_col, center)
            prev.drop(extra_features, inplace=True)
            s[center-counter, :] = prev.to_numpy()
        # next
        if i+counter < domain_df.shape[0]:
            post = domain_df.iloc[i+counter][temp_features]
            if p.timer_feature:
                timer_col = return_timer_feature(
                    curr, counter, post, timer_col, center)
            post.drop(extra_features, inplace=True)
            s[center+counter, :] = post.to_numpy()
        counter += 1

    curr.drop(extra_features, inplace=True)
    # center
    s[center, :] = curr.to_numpy()

    return s, timer_col


def add_temp_features(df):
    temp = ['created_at']
    if 'delivery_duration_sec' in df.columns:
        temp += ['delivery_duration_sec']
    copy = p.decision_features + temp
    return copy, temp


def stack_prev(s, i, domain_df, timer_col):
    '''
    for task 1, populates the stacked sample s by starting at the top of the stack with current row and walking
    backward with counter variable (from 0 to p.stack_length - 1).
    is built to add stack-dependent variables, such as timer_feature
    :param s: stacked_sample to populate (np.array)
    :param i: index of current element to help reference neighbors (int)
    :param domain_df: dataframe corresponding to current domain (pandas.dataFrame)
    :param timer_col: column of timer values to populate (np.array)
    :return: stacked sample (np.array)
    :return: timer_col, or the column of timer values to be concatenated (np.array)
    '''

    temp_features, extra_features = add_temp_features(domain_df)
    counter = 0

    curr = domain_df.iloc[i][temp_features]

    # iterating in descending order when reading the samples and populating the stack
    while counter < p.stack_length:
        if i-counter >= 0:
            slider = domain_df.iloc[i-counter][temp_features]
            if p.timer_feature:
                timer_col = return_timer_feature(curr, -counter, slider, timer_col)
            slider.drop(extra_features, inplace=True)
            s[p.stack_length-1-counter] = slider.to_numpy()
        counter += 1

    return s, timer_col


def stack_samples(df, domain):
    '''
    generates samples stacked on top of another in consecutive time steps to exploit temporal nature of data
    based on p.task, it follows either 2 schemes:
    1: given ith sample, single/consecutive samples from ith - p.stacking_length to ith sample (p.stacking_length >= 1)
    2: given ith sample, consecutive samples with ith sample at center, and previous and next samples, respectively, before and after
    :param df: dataset (pandas.dataFrame)
    :param domain: domain criteria used to stack samples in consecutive order (pandas.dataFrame)
    :return: stacked samples with shape (n_samples, p.stacking_length, p.decision_features) (np.array of np.arrays)
    :return: corresping labels samples with shape (n_samples, 1) (np.array of np.arrays)
    '''

    print("--- {} Decision features (stacking features could be added):\n {}".format(
        len(p.decision_features), p.decision_features))
    start_time = time.time()

    X = []
    y = []
    print("Task: {}".format(p.task))
    center = int((p.stack_length - 1) / 2)
    for v in df[domain].sort_values().unique():
        sub_X = []
        sub_y = []
        domain_df = df[df[domain] == v]
        domain_df.sort_values(by=['created_at'], ascending=[True], inplace=True)
        for i in range(domain_df.shape[0]):
            s = np.zeros((p.stack_length, len(p.decision_features)))  # stacked sample
            timer_col = np.zeros((p.stack_length, 1))  # additional column for time feature
            curr = domain_df.iloc[i]
            if p.task == 1:
                s, timer_col = stack_prev(s, i, domain_df, timer_col)
            else:
                s, timer_col = stack_prev_and_post(s, i, domain_df, timer_col, center)
            if p.stack_length > 1 and p.timer_feature:
                s = np.concatenate([s, timer_col], axis=1)
            sub_X.append(s)
            if 'delivery_duration_sec' in domain_df.columns:
                sub_y.append(np.array(curr['delivery_duration_sec']))
        X.append(np.array(sub_X))
        y.append(np.array(sub_y))

    X = np.array(X)
    y = np.array(y)

    if p.stacking_writing:
        print("WRITE ME STACK SAMPLES")
        exit()

    # TODO: a few assert statements

    print('--- Stacking for {} done in {} secs with final number of features: {} ---'.format(
        domain, time.time() - start_time, X[0].shape[-1]))

    return X, y
