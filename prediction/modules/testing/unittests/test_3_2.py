import unittest
from modules.main.feature_engineering import *
from modules.main.stacking import *
from modules.main.processing import *
from modules.testing.testing_methods import *
from modules.main import parameters as p

# EXAMPLE: another top example extracted from dataset, larger and involving other feature generation patterns

# Testing stacking

p.init({'debugging': '0', 'description': 'test'}, "unit_tests")
p.timer_feature = False  # means stacked sample have one more feature/column than dataframe
p.task = 1
p.stack_length = 1

data = u"""\
         market_id,    created_at, actual_delivery_time,  store_id,  subtotal,  total_onshift_dashers,  total_busy_dashers,  total_outstanding_orders,  estimated_store_to_consumer_driving_duration,  delivery_duration_sec,  week_day,  weekend,  day_of_month,  month,  hour, day_part,  1_hr_count_by_market_id, 1_hr_avg_subtotal_by_market_id,  24_hr_avg_time_by_market_id,  long_score_by_market_id,  short_score_by_market_id,  trip_to_customer_by_market_id,  1_hr_count_by_store_id, 1_hr_avg_subtotal_by_store_id,  24_hr_avg_time_by_store_id,  long_score_by_store_id,  short_score_by_store_id,  trip_to_customer_by_store_id
         1.0, 2015-01-21 15:22:03, 2015-01-21 16:17:43,      2966,     1058,                   2.0,                1.0,                       1.0,                                         463.0,                 3340.0,         2,        0,            21,      1,    15,        3,                        0,                        0.000000,                          0.0,                      0.0,                      0.00,                              0,                       0,                            0.0,                         0.0,                     0.0,                      0.0,                             0
         1.0, 2015-01-21 15:39:16, 2015-01-21 15:58:11,      5551,     1867,                   4.0,                1.0,                       1.0,                                         284.0,                 1135.0,         2,        0,            21,      1,    15,        3,                        1,                     1058.000000,                          0.0,                      0.0,                      0.00,                              0,                       0,                            0.0,                         0.0,                     0.0,                      0.0,                             0
         1.0, 2015-01-21 15:40:42, 2015-01-21 16:22:37,      602,      955,                    3.0,                2.0,                       2.0,                                         377.0,                 2515.0,         2,        0,            21,      1,    15,        3,                        2,                     1462.500000,                          0.0,                      0.0,                      0.00,                           2280,                       0,                            0.0,                         0.0,                     0.0,                      0.0,                             0
         1.0, 2015-01-21 15:41:33, 2015-01-21 16:35:14,      5591,     825,                    0.0,                0.0,                       0.0,                                         534.0,                 3221.0,         2,        0,            21,      1,    15,        3,                        3,                     1293.333333,                          0.0,                      0.0,                      0.00,                           4298,                       0,                            0.0,                         0.0,                     0.0,                      0.0,                             0
         1.0, 2015-01-21 15:46:41, 2015-01-21 16:22:11,      1000,     6400,                   4.0,                2.0,                       2.0,                                         231.0,                 2130.0,         2,        0,            21,      1,    15,        3,                        4,                     1176.250000,                          0.0,                      0.0,                      0.00,                            959,                       0,                            0.0,                         0.0,                     0.0,                      0.0,                             0
         1.0, 2015-01-21 16:06:37, 2015-01-21 16:37:01,      6367,     1600,                   1.0,                0.0,                       0.0,                                         312.0,                 1824.0,         2,        0,            21,      1,    16,        3,                        5,                     2221.000000,                          0.0,                      0.0,                      1.00,                           2079,                       0,                            0.0,                         0.0,                     0.0,                      0.0,                             0
         1.0, 2015-01-21 16:18:39, 2015-01-21 16:44:54,      2103,     2050,                   3.0,                2.0,                       2.0,                                         853.0,                 1575.0,         2,        0,            21,      1,    16,        3,                        6,                     2117.500000,                       2237.5,                      0.0,                      0.50,                           5338,                       0,                            0.0,                         0.0,                     0.0,                      0.0,                             0
         1.0, 2015-01-21 16:21:10, 2015-01-21 16:50:52,      1679,     1250,                   5.0,                3.0,                       3.0,                                         528.0,                 1782.0,         2,        0,            21,      1,    16,        3,                        7,                     2107.857143,                       2237.5,                      0.0,                      0.50,                           2274,                       0,                            0.0,                         0.0,                     0.0,                      0.0,                             0
         1.0, 2015-01-21 16:23:28, 2015-01-21 16:56:46,      3680,     1685,                   1.0,                1.0,                       1.0,                                         944.0,                 1998.0,         2,        0,            21,      1,    16,        3,                        7,                     2135.285714,                       2280.0,                      0.0,                      0.25,                           2256,                       0,                            0.0,                         0.0,                     0.0,                      0.0,                             0
         1.0, 2015-01-21 16:28:08, 2015-01-21 17:50:37,      977,      1607,                   1.0,                1.0,                       1.0,                                         679.0,                 4949.0,         2,        0,            21,      1,    16,        3,                        8,                     2079.000000,                       2280.0,                      0.0,                      0.25,                           2151,                       0,                            0.0,                         0.0,                     0.0,                      0.0,                             0
"""

stacked_data = [
    [
        [1, 2966, 1058, 2, 1, 1, 463, 2, 0, 21, 1, 15, 3, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 5551, 1867, 4, 1, 1, 284, 2, 0, 21, 1, 15, 3, 1, 0, 1058, 0, 0,
         0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 602, 955, 3, 2, 2, 377, 2, 0, 21, 1, 15, 3, 2, 0, 1462.5, 0, 0,
         0, 0, 0, 0, 0, 2280, 0]
    ],
    [
        [1, 5591, 825, 0, 0, 0, 534, 2, 0, 21, 1, 15, 3, 3, 0, 1293.33333333, 0, 0,
         0, 0, 0.0, 0, 0, 4298, 0]
    ],
    [
        [1, 1000, 6400, 4, 2, 2, 231, 2, 0, 21, 1, 15, 3, 4, 0, 1176.25, 0, 0,
         0, 0, 0.0, 0, 0, 959, 0]
    ],
]


class concrete_test_case(
    test_stacking(
        data, np.array([np.array(x) for x in stacked_data]),
        "market_id")):
    pass


if __name__ == '__main__':

    unittest.main()
