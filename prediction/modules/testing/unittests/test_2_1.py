import unittest
from modules.main.feature_engineering import *
from modules.main.stacking import *
from modules.main.processing import *
from modules.testing.testing_methods import *
from modules.main import parameters as p

# EXAMPLE: Top example extracted from dataset to familiarise with data and verify real data

# Testing features

p.init({'debugging': '0', 'description': 'test'}, "unit_tests")
data = u"""\
    market_id,       created_at,  actual_delivery_time, store_id, subtotal, total_onshift_dashers, total_busy_dashers, total_outstanding_orders, estimated_store_to_consumer_driving_duration, delivery_duration_sec, week_day, weekend, day_of_month, month, hour, day_part, 1_hr_count_by_market_id, 1_hr_avg_subtotal_by_market_id, 24_hr_avg_time_by_market_id, long_score_by_market_id,  short_score_by_market_id, trip_to_customer_by_market_id, 1_hr_count_by_store_id, 1_hr_avg_subtotal_by_store_id, 24_hr_avg_time_by_store_id, long_score_by_store_id, short_score_by_store_id, trip_to_customer_by_store_id
          1.0, 2015-01-21 20:35:27, 2015-01-21 21:01:26,     4149,     1750,                  18.0,               18.0,                     20.0,                                        259.0,                1559.0,        2,       0,           21,     1,   20,       4,                       0,                            0.0,                         0.0,                     0.0,                     0.00,                             0,                      0,                           0.0,                        0.0,                    0.0,                     0.0,                            0
          1.0, 2015-01-22 20:19:08, 2015-01-22 21:07:43,     4149,     2015,                  22.0,               13.0,                     13.0,                                        740.0,                2915.0,        3,       0,           22,     1,   20,       4,                       0,                            0.0,                         0.0,                     0.0,                     1.00,                             0,                      0,                           0.0,                        0.0,                    0.0,                     1.0,                            0
          1.0, 2015-01-24 02:01:20, 2015-01-24 02:46:13,     5058,     1800,                  27.0,               25.0,                     24.0,                                        424.0,                2693.0,        5,       1,           24,     1,    2,       0,                       0,                            0.0,                         0.0,                     0.0,                     0.50,                          2024,                      0,                           0.0,                        0.0,                    0.0,                     0.0,                            0
          1.0, 2015-01-26 02:09:48, 2015-01-26 02:47:22,     5058,     3800,                  21.0,               18.0,                     20.0,                                        530.0,                2254.0,        0,       0,           26,     1,    2,       0,                       0,                            0.0,                         0.0,                     0.0,                     0.33,                          2530,                      0,                           0.0,                        0.0,                    0.0,                     0.0,                            0
          1.0, 2015-01-27 02:12:36, 2015-01-27 03:02:24,     2841,     3620,                   5.0,                5.0,                      7.0,                                        205.0,                2988.0,        1,       0,           27,     1,    2,       0,                       0,                            0.0,                         0.0,                     0.0,                     0.25,                          1663,                      0,                           0.0,                        0.0,                    0.0,                     0.0,                            0
"""


class concrete_test_case(test_features(data)):
    pass


if __name__ == '__main__':

    unittest.main()
