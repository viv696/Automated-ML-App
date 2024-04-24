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
    
    def test_read_data_csv(self):
        # Test reading data from a CSV file
        df = read_data("dataset.csv")
        self.assertIsInstance(df, pd.DataFrame)

    # def test_read_data_excel(self):
    #     # Test reading data from an Excel file
    #     df = read_data("dataset.xlsx")
    #     self.assertIsInstance(df, pd.DataFrame)

    # def test_preprocess_data(self):
    #     # Test data preprocessing
    #     X, _, y, _ = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    #     df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    #     df['target'] = y

    #     x_train, _, y_train, _ = preprocess_data(df, target_column='target', scaler_type='standard')
    #     self.assertEqual(x_train.shape[0], y_train.shape[0])

    # def test_train_model(self):
    #     # Test model training
    #     X, _, y, _ = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    #     model = LogisticRegression()
    #     trained_model = train_model(X, y, model, model_name='logistic_regression')
    #     self.assertIsNotNone(trained_model)

    # def test_evaluate_model(self):
    #     # Test model evaluation
    #     X, _, y, _ = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    #     model = RandomForestClassifier()
    #     trained_model = train_model(X, y, model, model_name='random_forest')
    #     accuracy = evaluate_model(trained_model, X, y)
    #     self.assertGreaterEqual(accuracy, 0.0)
    #     self.assertLessEqual(accuracy, 1.0)


if __name__ == '__main__':
    unittest.main()
