import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from models.MultinomialNB import BORDER
from utilities.constants import JOHNNY_EXPERIMENT_THREE_PKL_VEC_DIR, JOHNNY_EXPERIMENT_THREE_PKL_DIR
from utilities.utility import pickle_model
from sklearn.metrics import root_mean_squared_error


class KNNRegressor(KNeighborsRegressor):
    def __init__(self):
        super().__init__()

    @staticmethod
    def perform_experiment_three_training(file_path: str):
        """
        Performs experiment 3 but in training mode.

        @attention: Experiment Details
            This experiment aims to differ from experiment 1 and 2
            by using KNN regressor instead of the Multinomial NB
            (classifier) to perform sentimental analysis, in hopes
            it yields better results.

        @param file_path:
            A string representing the file path
            of test JSON file

        @return: None
        """
        print("[+] Now performing Johnny's experiment 3 in training mode...")

        # Load the Preprocessed Data
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
            df = pd.DataFrame(data)

        # Define features and target labels
        X = df['text']
        y = df[['stars', 'useful', 'funny', 'cool']]  # Target variables

        # Split dataset (training 70%, validation 15%, testing 15%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test,
                                                                      test_size=0.5, random_state=42)

        # Vectorize text data
        vectorizer = TfidfVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_validation_vectorized = vectorizer.transform(X_validation)
        X_test_vectorized = vectorizer.transform(X_test)

        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
        }

        # Perform hyperparameter tuning with validaton set
        grid_search = GridSearchCV(KNeighborsRegressor(n_jobs=2), param_grid, cv=5, n_jobs=2)
        grid_search.fit(X_validation_vectorized, y_validation)
        print("[+] Best hyperparameters for KNN regressor: ", grid_search.best_params_)

        # Train with best hyperparameters
        best_knn_regressor = grid_search.best_estimator_
        best_knn_regressor.fit(X_train_vectorized, y_train)

        # Evaluate and test the regressor's performance
        y_pred = best_knn_regressor.predict(X_test_vectorized)

        # Compute evaluation metrics for each target variable individually
        for i, target_name in enumerate(y.columns):
            mse = mean_squared_error(y_test[target_name], y_pred[:, i])
            mae = mean_absolute_error(y_test[target_name], y_pred[:, i])
            rmse = root_mean_squared_error(y_test[target_name], y_pred[:, i])
            r_squared = r2_score(y_test[target_name], y_pred[:, i])
            explained_variance = explained_variance_score(y_test[target_name], y_pred[:, i])
            medae = median_absolute_error(y_test[target_name], y_pred[:, i])

            print(BORDER)
            print(f"[+] MSE for {target_name}: {mse}")
            print(f"[+] MAE for {target_name}: {mae}")
            print(f"[+] RMSE for {target_name}: {rmse}")
            print(f"[+] R-squared for {target_name}: {r_squared}")
            print(f"[+] Explained Variance for {target_name}: {explained_variance}")
            print(f"[+] Median Absolute Error for {target_name}: {medae}")
            print(BORDER)

        # Pickle the final models
        pickle_model(vectorizer, save_path=JOHNNY_EXPERIMENT_THREE_PKL_VEC_DIR)
        pickle_model(best_knn_regressor, save_path=JOHNNY_EXPERIMENT_THREE_PKL_DIR)

    @staticmethod
    def perform_experiment_three_inference(file_path: str):
        """
        Performs experiment 3, but in inference mode.

        @param file_path:
            A string representing the file path
            of test JSON file

        @return: None
        """
        print("[+] Now performing Johnny's experiment 3 in inference mode...")

        # Load the Preprocessed Data
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
            df = pd.DataFrame(data)

        # Get test data
        X_test = df['text']
        y_test = df[['stars', 'useful', 'funny', 'cool']]

        # Load vectorizer and transform test data
        with open(JOHNNY_EXPERIMENT_THREE_PKL_VEC_DIR, 'rb') as f:
            vectorizer = pickle.load(f)
            X_test_vectorized = vectorizer.transform(X_test)

        # Load the model
        with open(JOHNNY_EXPERIMENT_THREE_PKL_DIR, 'rb') as f:
            classifier = pickle.load(f)

        # Predict on test data
        y_pred_test = classifier.predict(X_test_vectorized)

        # Compute evaluation metrics for each target variable individually
        for i, target_name in enumerate(y_test.columns):
            true_values = y_test[target_name].values
            predicted_values = y_pred_test[:, i]

            mse = np.mean((true_values - predicted_values) ** 2)
            mae = np.mean(np.abs(true_values - predicted_values))
            rmse = np.sqrt(mse)
            r_squared = 1 - np.sum((true_values - predicted_values) ** 2) / np.sum(
                (true_values - np.mean(true_values)) ** 2)
            explained_variance = 1 - np.var(true_values - predicted_values) / np.var(true_values)
            medae = np.median(np.abs(true_values - predicted_values))

            print(BORDER)
            print(f"[+] MSE for {target_name}: {mse}")
            print(f"[+] MAE for {target_name}: {mae}")
            print(f"[+] RMSE for {target_name}: {rmse}")
            print(f"[+] R-squared for {target_name}: {r_squared}")
            print(f"[+] Explained Variance for {target_name}: {explained_variance}")
            print(f"[+] Median Absolute Error for {target_name}: {medae}")
            print(BORDER)
