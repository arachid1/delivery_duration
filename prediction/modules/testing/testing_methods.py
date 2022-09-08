import unittest
from io import StringIO
from modules.main.feature_engineering import *
from modules.main.stacking import *
from modules.main.processing import *
from modules.main import parameters as p
from pandas.util.testing import assert_frame_equal
np.set_printoptions(suppress=True)


# TODO: update comments
def setup(data, search_space_data):
    '''
    helps read the 100% correct, written dataframe for our test cases
    and set up the configs/environnments for the unit tests
    :param data: unicode string to search space (str)
    :param search_space_data: unicode string to additional/different search space (dict)
    :return: the dataframe that was read
    '''

    separator = r",\s*"
    df = pd.read_csv(StringIO(data), index_col=False, sep=separator, engine='python')
    df = clean_data(df, ['total_onshift_dashers',
                    'total_busy_dashers', 'total_outstanding_orders'])
    names = ['created_at']
    if 'actual_delivery_time' in df.columns:
        names += ['actual_delivery_time']
    df = process_timestamps(df, names)

    p.feature_writing = True

    if search_space_data is None:
        # default configs, TODO: switch set up to inside each test file but good for now!
        p.decision_features += p.time_features
        p.decision_features += p.domain_features
        p.domain_settings = [
            ("{}_hr_count_by_".format(1),
             return_count, 1, 'default', None),
            ("{}_hr_avg_subtotal_by_".format(1),
             return_avg_subtotal, 1, 'default', None),
            ("{}_hr_avg_time_by_".format(24),
             return_avg_time, 24, 'default', None),
            ("long_score_by_", return_long_score, 1460, 'default', None),
            ("short_score_by_", return_short_score, 1460, 'default', None),
            ("trip_to_customer_by_", return_trip_to_customer, 1460, 'default', None)]

    search_spaces = df

    if search_space_data is not None:
        search_spaces = pd.read_csv(
            StringIO(search_space_data),
            index_col=False, sep=separator, engine='python')
        search_spaces = clean_data(
            search_spaces,
            ['total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders'])
        search_spaces = process_timestamps(search_spaces, ['created_at', 'actual_delivery_time'])
        search_spaces = {'historical': search_spaces}
    return df, search_spaces


def test_features(data, search_space_data=None):
    '''
    is given a 100% correct dataframe named data
    - first test (on individual funcs) compares each individual dataframe element with the function call
    that generated it
    - second test (on gen) doesn't call individual functions but generates the entire dictionary,
    and compares its value one by one against the dataframe
    - third test (on add_domain_features) created a copy data frame, drops its domain features, and
    regenerates them to compare the output dataframe with the original one
    :param data: dataset (pandas.dataFrame)
    :param search_space_data: unicode string to additional/different search space (dict)
    :returns: a unnittest test cast
    '''

    class individual_test_case(unittest.TestCase):

        @classmethod
        def setUpClass(self):
            self.data = data
            self.search_space_data = search_space_data

        # comparing correct written dataframe elements one by one with individual functions calls (determined by p.domain_settings)
        def test_individual_funcs(self):
            df, search_spaces = setup(self.data, self.search_space_data)
            for i in range(df.shape[0]):
                row = df.iloc[i]
                for domain in p.domain_columns:
                    domain_df = df.loc[df[domain] == row[domain]]
                    search_df = domain_df
                    for y, (feature_name, func, n_hours, space_key, space_names) in enumerate(p.domain_settings):
                        if search_spaces is not None and space_key is not 'default':
                            if space_key is 'add':
                                for space_name in space_names:
                                    space = search_spaces[space_name]
                                    space = space[domain_df.columns.intersection(space.columns)]
                                    search_df = pd.concat(
                                        [search_df, space],
                                        ignore_index=True, axis=0)
                            else:
                                search_df = search_spaces[space_key]

                        search_df = search_df.loc[search_df[domain] == row[domain]]
                        feature_name = str(feature_name + domain)
                        truth_value = row[feature_name]
                        generated = func(search_df, row, n_hours)
                        np.testing.assert_almost_equal(
                            truth_value,
                            generated,
                            5)

        # testing 'gen' function which returns all feature lists for a given domain
        def test_gen(self):
            df, search_spaces = setup(self.data, self.search_space_data)
            for domain in p.domain_columns:
                df.sort_values(by=[domain, 'created_at'], ascending=[True, True], inplace=True)
                all = gen(df, domain, search_spaces=search_spaces)
                for (k, v) in all.items():
                    col = df[k].tolist()
                    self.assertEqual(len(col), len(v))
                    for i in range(len(col)):
                        truth_value = col[i]
                        generated = v[i]
                        np.testing.assert_almost_equal(truth_value, generated, 5)

        # testing 'add_domain_features' that returns the data frame with new domain features with IO system
        def test_add_domain_features(self):

            df, search_spaces = setup(self.data, self.search_space_data)

            df_copy = df.copy()
            df_copy.drop(p.domain_features, axis=1, inplace=True)
            generated_df = add_domain_features(
                df_copy, "unittest_data/unittest_", search_spaces=search_spaces)

            # putting them in the same order and conv. indexes form range object to numpy
            generated_df.sort_values(by=['created_at'], ascending=[True], inplace=True)
            df.sort_values(by=['created_at'], ascending=[True], inplace=True)
            df = df.reset_index(drop=True)
            generated_df = generated_df.reset_index(drop=True)

            try:
                assert_frame_equal(df, generated_df)
            except AssertionError as e:
                raise self.failureException("assert_frame_equal failed.") from e

        # TODO:
        # test add_features here instead of in test_stacking,
        # which should complete every function by testing 'add_time_features'
        # (not urgent bc add_time_features uses pandas function calls but for completeness)

    return individual_test_case

# TODO: update stacking test to take in different search space?


def test_stacking(data, stacked_data, domain_id):
    '''
    - is given a 100% correct dataframe named data and its corresponding stacked_data
    one big test but verifies 2 things:
    - first subtest (add_features) drops time and domain features and regenerates them correctly
    - second subtest (stack_samples) generates a stack by domain_id from data <-> compares it to stacked_data
    :param data: dataset (pandas.dataFrame)
    :param stacked_data: pre-written data (np.array of np.arrays)
    :param domain_id: domain to stack samples by (string)
    :returns: a unnittest test cast
    '''
    class individual_test_case(unittest.TestCase):

        @classmethod
        def setUpClass(self):
            self.data = data
            self.stacked_data = stacked_data
            self.domain_id = domain_id

        # tests not only stacking but add_features as well
        def test_add_features_and_stacking(self):
            df, _ = setup(self.data, None)
            # dropping features
            df.drop(p.time_features, axis=1, inplace=True)
            df.drop(p.domain_features, axis=1, inplace=True)
            df = add_features(df, "unnitest_")  # adding them all
            X, y = stack_samples(df, self.domain_id)
            X = [x for batch in X for x in batch]  # flatten the domains for orders to match
            for i, truth_value in enumerate(self.stacked_data):
                generated = X[i]
                try:
                    np.testing.assert_almost_equal(truth_value, generated, 5)
                except AssertionError as e:
                    raise self.failureException("assert_frame_equal failed.") from e

    return individual_test_case
