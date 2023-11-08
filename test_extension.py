import unittest

import numpy as np
from sklearn.dummy import DummyRegressor

from extension_dummy_regressor import ExtensionDummyRegressor


class ExtensionDummyRegressorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X_1 = np.array([1.0, 2.0, 3.0])
        cls.y_1 = np.array([0.5, 1.3, -0.8])
        cls.X_2 = np.array([[1, 5], [2, 1], [3, 4]])
        cls.y_2 = np.array([5, 3, -8])
        cls.X_3 = np.array([[1, 5], [2, 1], [3, 4]])
        cls.y_3 = np.array(
            [[0.5, 1.3, -0.8], [0.4, 1.4, -0.1], [0.8, 1.6, -0.2]]
        )

    def test_fracsum_strategy(self) -> None:
        '''Результат стратегии fracsum соответствует ожидаемому.'''

        ext_dummy_regr = ExtensionDummyRegressor(strategy='fracsum')
        ext_dummy_regr.fit(self.X_1, self.y_1)
        self.assertEqual(ext_dummy_regr.constant_, np.sum(self.y_1 % 1))
        ext_dummy_regr = ExtensionDummyRegressor(strategy='fracsum')
        ext_dummy_regr.fit(self.X_2, self.y_2)
        self.assertEqual(ext_dummy_regr.constant_, np.sum(self.y_2 % 1))

    def test_extension_dummy_regressor_mean(self):
        '''Результат стратегии mean дополненного класса DummyRegressor.

        Дополненный класс DummyRegressor обрабатывает стратегию mean так же,
        как изначальный класс DummyRegressor.
        '''

        ext_dummy_regr_mean = ExtensionDummyRegressor(strategy='mean')
        ext_dummy_regr_mean.fit(self.X_1, self.y_1)
        dummy_regr_mean = DummyRegressor(strategy='mean')
        dummy_regr_mean.fit(self.X_1, self.y_1)
        self.assertEqual(
            ext_dummy_regr_mean.constant_, dummy_regr_mean.constant_
        )
        ext_dummy_regr_mean = ExtensionDummyRegressor(strategy='mean')
        ext_dummy_regr_mean.fit(self.X_2, self.y_2)
        dummy_regr_mean = DummyRegressor(strategy='mean')
        dummy_regr_mean.fit(self.X_2, self.y_2)
        self.assertEqual(
            ext_dummy_regr_mean.constant_, dummy_regr_mean.constant_
        )

    def test_extension_dummy_regressor_median(self):
        '''Результат стратегии median дополненного класса DummyRegressor.

        Дополненный класс DummyRegressor обрабатывает стратегию median так же,
        как изначальный класс DummyRegressor.
        '''

        ext_dummy_regr_median = ExtensionDummyRegressor(strategy='median')
        ext_dummy_regr_median.fit(self.X_1, self.y_1)
        dummy_regr_median = DummyRegressor(strategy='median')
        dummy_regr_median.fit(self.X_1, self.y_1)
        self.assertEqual(
            ext_dummy_regr_median.constant_, dummy_regr_median.constant_
        )
        ext_dummy_regr_median = ExtensionDummyRegressor(strategy='median')
        ext_dummy_regr_median.fit(self.X_2, self.y_2)
        dummy_regr_median = DummyRegressor(strategy='median')
        dummy_regr_median.fit(self.X_2, self.y_2)
        self.assertEqual(
            ext_dummy_regr_median.constant_, dummy_regr_median.constant_
        )

    def test_multidimensional_regression(self):
        '''Обработка многомерных значений.'''

        multi_ext_dummy_regr = ExtensionDummyRegressor(strategy='fracsum')
        multi_ext_dummy_regr.fit(self.X_3, self.y_3)
        self.assertEqual(multi_ext_dummy_regr.constant_, np.sum(self.y_3 % 1))


if __name__ == '__main__':
    unittest.main()
