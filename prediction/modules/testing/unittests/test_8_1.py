import unittest
from modules.main.feature_engineering import *
from modules.main.stacking import *
from modules.main.processing import *
from modules.testing.testing_methods import *
from modules.main import parameters as p

# EXAMPLE: adjusting example from test 4 to test different search space

# Testing features with different search space (more complex now, see 6_1)


p.init({'debugging': '0', 'description': 'test'}, "unit_tests")
p.domain_settings = [
    ("{}_hr_count_by_".format(1),
     return_count, 1, 'default', None),
    ("{}_hr_avg_subtotal_by_".format(1),
     return_avg_subtotal, 1, 'default', None),
    ("long_score_by_", return_long_score, 1460, 'historical', None),
    ("short_score_by_", return_short_score, 1460, 'historical', None),
    ("trip_to_customer_by_", return_trip_to_customer, 1460, 'historical', None)]
p.decision_features += p.time_features
p.domain_features.remove('24_hr_avg_time_by_market_id')
p.domain_features.remove('24_hr_avg_time_by_store_id')
p.decision_features += p.domain_features


# replicating test data without actual_delivery_time or delivey_duration_sec
data = u"""\
         market_id,    created_at, store_id,  subtotal,  total_onshift_dashers,  total_busy_dashers,  total_outstanding_orders,  estimated_store_to_consumer_driving_duration,  week_day,  weekend,  day_of_month,  month,  hour, day_part,  1_hr_count_by_market_id, 1_hr_avg_subtotal_by_market_id,  long_score_by_market_id,  short_score_by_market_id,  trip_to_customer_by_market_id,  1_hr_count_by_store_id, 1_hr_avg_subtotal_by_store_id,  long_score_by_store_id,  short_score_by_store_id,  trip_to_customer_by_store_id
         1.0, 2015-01-25 10:35:00,       9,     2000,                   2.0,                1.0,                       1.0,                                       300.0,               3,        0,           15,       1,     10,        1,                      1,                         1200.0,                       0.0,                     0.67,                            1650,                       1,                         1200.0,                    0.0,                      1.0,                             0
         1.0, 2015-01-25 10:20:00,       9,     1200,                   2.0,                1.0,                       1.0,                                       300.0,               3,        0,           15,       1,     10,        1,                      0,                            0.0,                       0.0,                     0.67,                            1650,                       0,                            0.0,                    0.0,                      1.0,                             0
         2.0, 2015-01-25 10:15:00,       15,    2000,                   2.0,                1.0,                       1.0,                                      1000.0,               3,        0,           15,       1,     10,        1,                      1,                          500.0,                       0.5,                      0.5,                            2700,                       1,                          500.0,                    0.0,                      0.0,                             0
         2.0, 2015-01-25 10:00:00,       15,     500,                   2.0,                1.0,                       1.0,                                       500.0,               3,        0,           15,       1,     10,        1,                      0,                            0.0,                       0.5,                      0.5,                            2700,                       0,                            0.0,                    0.0,                      0.0,                             0
         2.0, 2015-01-16 10:30:00,       15,    2000,                   2.0,                1.0,                       1.0,                                       500.0,               3,        0,           15,       1,     10,        1,                      1,                         1000.0,                       0.5,                      0.5,                            2700,                       1,                         1000.0,                    0.0,                      0.0,                             0
         2.0, 2015-01-16 10:00:00,       15,    1000,                   2.0,                1.0,                       1.0,                                       500.0,               3,        0,           15,       1,     10,        1,                      0,                            0.0,                       0.5,                      0.5,                            2700,                       0,                            0.0,                    0.0,                      0.0,                             0
"""
search_space_data = u"""\
         market_id,    created_at, actual_delivery_time,  store_id,  subtotal,  total_onshift_dashers,  total_busy_dashers,  total_outstanding_orders,  estimated_store_to_consumer_driving_duration,  delivery_duration_sec,  week_day,  weekend,  day_of_month,  month,  hour, day_part,  1_hr_count_by_market_id, 1_hr_avg_subtotal_by_market_id,  long_score_by_market_id,  short_score_by_market_id,  trip_to_customer_by_market_id,  1_hr_count_by_store_id, 1_hr_avg_subtotal_by_store_id,  long_score_by_store_id,  short_score_by_store_id,  trip_to_customer_by_store_id
         2.0, 2015-01-16 8:00:00, 2015-01-16 9:00:00,      30,    1200,                   2.0,                1.0,                       1.0,                                         500.0,                 3600.0,               3,        0,           15,       1,      8,        1,                        1,                   1500.000000,                        0.33,                     0.67,                            2400,                       1,                         1500.0,                        0.33,                     0.67,                          2400
         2.0, 2015-01-16 7:00:00, 2015-01-16 7:30:00,      30,    1500,                   2.0,                1.0,                       1.0,                                         500.0,                 1800.0,               3,        0,           15,       1,      7,        1,                        1,                   2000.000000,                         0.5,                      0.5,                            2700,                       1,                         2000.0,                         0.5,                      0.5,                          2700
         2.0, 2015-01-16 6:00:00, 2015-01-16 6:30:00,      30,    2000,                   2.0,                1.0,                       1.0,                                         500.0,                 1800.0,               3,        0,           15,       1,      6,        1,                        0,                      0.000000,                         1.0,                      0.00,                              0,                       0,                            0.0,                         1.0,                      0.0,                             0
         2.0, 2015-01-15 6:00:00, 2015-01-15 7:00:00,      30,    2000,                   2.0,                1.0,                       1.0,                                         500.0,                 3600.0,               3,        0,           15,       1,      6,        1,                        0,                      0.000000,                         0.0,                      0.00,                              0,                       0,                            0.0,                         0.0,                      0.0,                             0
         1.0, 2015-01-01 7:15:00, 2015-01-01 8:00:00,      1,     1000,                   2.0,                1.0,                       1.0,                                         100.0,                 2700.0,               3,        0,           1,        1,      7,        1,                        1,                   1000.000000,                         0.0,                      1.00,                           1200,                       1,                         1000.0,                         0.0,                      1.0,                          1200
         1.0, 2015-01-01 7:00:00, 2015-01-01 7:20:00,      1,     1000,                   2.0,                1.0,                       1.0,                                         100.0,                 1200.0,               3,        0,           1,        1,      7,        1,                        1,                   1000.000000,                         0.0,                      1.00,                              0,                       1,                         1000.0,                         0.0,                      1.0,                             0
         1.0, 2015-01-01 6:00:00, 2015-01-01 6:30:00,      1,     1000,                   2.0,                1.0,                       1.0,                                         200.0,                 1800.0,               3,        0,           1,        1,      6,        1,                        0,                      0.000000,                         0.0,                      0.00,                              0,                       0,                            0.0,                         0.0,                      0.0,                             0
         0.0, 2015-01-01 1:00:00, 2015-01-01 1:20:00,      9,      200,                   2.0,                1.0,                       1.0,                                         200.0,                 1800.0,               3,        0,           1,        1,      6,        1,                        0,                      0.000000,                         0.0,                      0.00,                              0,                       0,                            0.0,                         0.0,                      0.0,                             0
"""


class concrete_test_case(test_features(data, search_space_data)):
    pass


if __name__ == '__main__':

    unittest.main()
