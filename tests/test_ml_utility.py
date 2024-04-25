import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from AutoML.src.ml_utility import read_data, preprocess_data, train_model, evaluate_model


class TestMLUtility(unittest.TestCase):
    
    def setUp(self):
        # Sample data for testing
        data = {
            'CryoSleep': [False, False, False, False, False, False, False, True, False],
            'Age': [39.0, 24.0, 58.0, 33.0, 16.0, 44.0, 26.0, 28.0, 35.0],
            'VIP': [False, False, True, False, False, False, False, False, False],
            'RoomService': [0.0, 109.0, 43.0, 0.0, 303.0, 0.0, 42.0, 0.0, 0.0],
            'FoodCourt': [0.0, 9.0, 3576.0, 1283.0, 70.0, 483.0, 1539.0, 0.0, 785.0],
            'ShoppingMall': [0.0, 25.0, 0.0, 371.0, 151.0, 0.0, 3.0, 0.0, 17.0],
            'Spa': [0.0, 549.0, 6715.0, 3329.0, 565.0, 291.0, 0.0, 0.0, 216.0],
            'VRDeck': [0.0, 44.0, 49.0, 193.0, 2.0, 0.0, 0.0, None, 0.0],
            'Transported': [False, True, False, False, True, True, True, True, True]
        }
        self.df = pd.DataFrame(data)
        
    def test_read_data_csv(self):
        # Test reading data from a CSV file
        df = read_data("dataset.csv")
        self.assertIsInstance(df, pd.DataFrame)

    def test_read_data_excel(self):
        # Test reading data from an Excel file
        df = read_data("dataset.xlsx")
        self.assertIsInstance(df, pd.DataFrame)

    def test_preprocess_data(self):
        self.setUp()
        # Test preprocessing data
        x_train, x_test, y_train, y_test = preprocess_data(self.df, target_column='Transported', scaler_type='standard')
        
        # Check if the returned variables are of the correct type
        self.assertIsInstance(x_train, pd.DataFrame)
        self.assertIsInstance(x_test, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)

        # Check if the shape of the training and test data is correct
        self.assertEqual(x_train.shape[0], 7)  # 7 rows for training data
        self.assertEqual(x_test.shape[0], 2)   # 2 rows for test data
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
    def test_train_model(self):
        # Test model training
        self.test_preprocess_data()
        model = LogisticRegression()
        trained_model = train_model(self.x_train, self.y_train, model, model_name='logistic_regression')
        self.assertIsNotNone(trained_model)

    def test_evaluate_model(self):
        # Test model evaluation
        self.test_preprocess_data()
        model = RandomForestClassifier()
        trained_model = train_model(self.x_train, self.y_train, model, model_name='random_forest')
        accuracy = evaluate_model(trained_model, self.x_test,self.y_test)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == '__main__':
    unittest.main()
