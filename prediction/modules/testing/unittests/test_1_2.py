import unittest
from modules.main.feature_engineering import *
from modules.main.stacking import *
from modules.main.processing import *
from modules.testing.testing_methods import *
from modules.main import parameters as p

# EXAMPLE: custom medium size dataset with multiple orders markets, stores, etc, englobing every feature and rule!
# this is my first example so a bit messy, see test 2 for improved results

# Testing stacking

p.init({'debugging': '0', 'description': 'test'}, "unit_tests")
p.timer_feature = True  # means stacked sample have one more feature/column than dataframe
p.task = 1
p.stack_length = 1

data = u"""\
    market_id, created_at, actual_delivery_time, store_id, subtotal, total_onshift_dashers, total_busy_dashers, total_outstanding_orders, estimated_store_to_consumer_driving_duration, delivery_duration_sec, hour, week_day, day_of_month, month, 1_hr_count_by_market_id, 1_hr_avg_subtotal_by_market_id, 24_hr_avg_time_by_market_id, long_score_by_market_id, short_score_by_market_id, trip_to_customer_by_market_id, 1_hr_count_by_store_id, 1_hr_avg_subtotal_by_store_id, 24_hr_avg_time_by_store_id, long_score_by_store_id, short_score_by_store_id, trip_to_customer_by_store_id
    1, 2015-02-06 22:20:02, 2015-02-06 22:50:02, 1845, 2500, 8, 4, 5, 500,  1800,   21, 5, 6, 2,    2, 3250.0, 0.0,   0.4, 0.4,   2911,      1, 5000.0, 0.0,   0.0, 1.0,      1443
    1, 2015-02-06 22:20:01, 2015-02-06 22:50:01, 1845, 5000, 4, 2, 5, 200,  1800,   21, 2, 6, 2,     1, 1500.0, 0.0,   0.4, 0.4,   3356,      0, 0.0, 0.0,      0.0, 1.0,      1200
    1, 2015-02-06 22:20:00, 2015-02-06 22:50:00, 1500, 1500, 5, 1, 5, 1500,  1800,   21, 4, 6, 2,    0, 0.0, 0.0,      0.4, 0.4,    2695,       0, 0.0, 0.0,       0.67, 0.0,   4300
    1, 2015-02-06 20:20:00, 2015-02-06 20:40:00, 1845, 3000, 9, 8, 7, 200,   1200,   19, 2, 6, 2,    0, 0.0, 0.0,     0.5, 0.25,    4224,     0, 0.0, 0.0,     0.0, 1.0,        0
    1, 2015-02-02 20:20:20, 2015-02-02 20:30:20, 1845, 3000, 1, 1, 1, 5000,  600,   19, 6, 2, 2,    0, 0.0, 0.0,      0.67, 0.0,      4300,      0, 0.0, 0.0,      0.0, 0.0,      0
    1, 2015-02-01 20:20:00, 2015-02-01 20:55:00, 1500, 3000, 4, 2, 5, 100,   2100,   19, 1, 1, 2,    0, 0.0, 5400,      1.0, 0.0,       5400,     0, 0.0, 5400,      1.0, 0.0,      5400
    1, 2015-02-01 17:00:00, 2015-02-01 18:00:00, 1500, 6000, 4, 2, 5, 100,   3600,   19, 1, 1, 2,    0, 0.0, 0.0,       1.0, 0.0,       0,    0, 0.0, 0.0,       1.0, 0.0,          0
    1, 2015-02-01 12:00:00, 2015-02-01 14:00:00, 1500, 2000, 4, 2, 5, 100,  7200,   19, 1, 1, 2,    0, 0.0, 0.0,        0.0, 0.0,       0,    0, 0.0, 0.0,        0.0, 0.0,         0
    0, 2015-01-30 8:30:00, 2015-01-30 10:00:00, 3000, 500, 4, 4, 5, 100,  5400,   8, 1, 30, 1,    1, 500, 0.0,        0.0, 0.0,       3600,    1, 500, 0.0,       0.0, 0.0,     3600
    0, 2015-01-30 8:00:00, 2015-01-30 9:00:00, 3000, 500, 4, 4, 5, 100,  3600,   8, 1, 30, 1,    0, 0.0, 0.0,        0.0, 0.0,       0,    0, 0.0, 0.0,       0.0, 0.0,          0
    0, 2015-01-30 6:30:00, 2015-01-30 9:00:00, 3000, 500, 10, 4, 2, 500,  13500,   6, 1, 30, 1,    0, 0.0, 0.0,        0.0, 0.0,       0,    0, 0.0, 0.0,        0.0, 0.0,         0

"""

stacked_data = [
    [
        [0, 3000, 500, 10, 4, 2, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0]
    ],
    [
        [0, 3000, 500, 4, 4, 5, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0]
    ],
    [
        [0, 3000, 500, 4, 4, 5, 100, 1, 1, 500, 500, 0, 0, 0, 0,
            0, 0, 3600, 3600]
    ],
    [
        [1, 1500, 2000, 4, 2, 5, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0]
    ],
    [
        [1, 1500, 6000, 4, 2, 5, 100, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
         0]
    ],
    [
        [1, 1500, 3000, 4, 2, 5, 100, 0, 0, 0, 0, 5400, 5400, 1, 0, 1,
         0, 5400, 5400]
    ],
    [
        [1, 1845, 3000, 1, 1, 1, 5000, 0, 0, 0, 0, 0, 0, 0.67, 0, 0, 0.0, 4300.0, 0]
    ]
]


class concrete_test_case(
    test_stacking(
        data, np.array([np.array(x) for x in stacked_data]),
        "market_id")):
    pass


if __name__ == '__main__':

    unittest.main()
