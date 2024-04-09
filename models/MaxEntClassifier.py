import json
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from models.MultinomialNB import BORDER
from utilities.constants import JOHNNY_EXPERIMENT_THREE_PKL_VECTORIZE_DIR, JOHNNY_EXPERIMENT_THREE_PKL_DIR, \
    JOHNNY_MAX_ENT_BASE_EXPERIMENT_VEC_PKL_DIR, JOHNNY_MAX_ENT_BASE_EXPERIMENT_PKL_DIR
from utilities.utility import pickle_model


class MaxEntClassifier(LogisticRegression):
    def __init__(self):
        super().__init__()

    @staticmethod
    def perform_base_experiment_training(file_path: str):
        print("[+] Now performing Maximum Entropy (Base Experiment) in training mode...")

        # Load the Preprocessed Data
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
            df = pd.DataFrame(data)

        # Feature Extraction
        X = df['text']
        y = df[['stars', 'useful', 'funny', 'cool']]  # Target variables

        # Split dataset (training 70%, validation 15%, testing 15%)
        X_train, X_test, y_train, y_test = train_test_split(X, y,  # (70% train, 30% test)
                                                            test_size=0.3,
                                                            random_state=42)

        # 50% temp, 50% split between validation and test
        X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test,
                                                                      test_size=0.5, random_state=42)

        # Vectorize text data (N-grams & Bag of Words)
        ngram_range = (1, 2)
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_validation_vectorized = vectorizer.transform(X_validation)
        X_test_vectorized = vectorizer.transform(X_test)

        # Save and pickle the vectorizer
        pickle_model(vectorizer, save_path=JOHNNY_MAX_ENT_BASE_EXPERIMENT_VEC_PKL_DIR)

        # Train classifiers and perform hyperparameter tuning per target variable
        for target in y_train.columns:
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(X_train_vectorized, y_train[target])

            y_pred_test = classifier.predict(X_test_vectorized)

            # Save and pickle the classifier
            pickle_model(classifier, save_path=JOHNNY_MAX_ENT_BASE_EXPERIMENT_PKL_DIR, target=target)

            print(BORDER)
            if target != 'stars':
                mse = mean_squared_error(y_test[target], y_pred_test)
                print(f"Mean Squared Error (MSE) for Target Variable '{target}' on Test Set: {mse}")
            print(f"Classification Report for Target Variable ('{target}') on Test Set: ")
            print(classification_report(y_test[target], y_pred_test, zero_division=1))
            print(BORDER)

    @staticmethod
    def perform_base_experiment_inference(file_path: str):
        print("[+] Now performing Maximum Entropy (Base Experiment) in inference mode...")

        # Load JSON test dataset
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
            test_df = pd.DataFrame(data)

        # Extract labels (target variables)
        y = test_df[['stars', 'useful', 'funny', 'cool']]

        # Load vectorizer model
        with open(JOHNNY_MAX_ENT_BASE_EXPERIMENT_VEC_PKL_DIR, 'rb') as f:
            vectorizer = pickle.load(f)

        # Vectorize the test data
        X_test_vectorized = vectorizer.transform(test_df['text'])

        # Evaluate each classifier
        for target in y.columns:
            with open(JOHNNY_MAX_ENT_BASE_EXPERIMENT_PKL_DIR.format(target), 'rb') as f:
                classifier = pickle.load(f)
                y_pred_test = classifier.predict(X_test_vectorized)

                print(BORDER)
                if target != 'stars':
                    mse = mean_squared_error(test_df[target], y_pred_test)
                    print(f"Mean Squared Error (MSE) for Target Variable '{target}' on Test Set: {mse}")
                print(f"Classification Report for Target Variable ('{target}') on Test Set: ")
                print(classification_report(test_df[target], y_pred_test, zero_division=1))
                print(BORDER)

    @staticmethod
    def perform_experiment_three_training(file_path: str):
        """
        Performs experiment 3 using training mode.

        @attention: Experiment Details
            This experiment is an extension of experiment 2,
            but using Maximum Entropy (Max Ent) model.

            Hypothesis: Maximum Entropy (MaxEnt) performs better
            at predicting and classifying sentiment than Multinomial
            Naive Bayes algorithm.

        @param file_path:
            A string representing the file path
            of test JSON file

        @return: None
        """
        print("[+] Now performing Johnny's experiment 3 in training mode...")

        def stars_to_sentiment(stars: int):
            if stars <= 2:
                return 'negative'
            elif stars == 3:
                return 'neutral'
            else:
                return 'positive'

        def categorize_value(value: int, thresholds: dict):
            for category, threshold in thresholds.items():
                if value <= threshold:
                    return category
            return None

        # Load the Preprocessed Data
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
            df = pd.DataFrame(data)

        # Categorize 'useful', 'funny', and 'cool' columns
        useful_thresholds = {'not useful': df['useful'].quantile(0.25),
                             'neutral': df['useful'].quantile(0.5),
                             'very useful': df['useful'].max()}
        funny_thresholds = {'not funny': df['funny'].quantile(0.25),
                            'neutral': df['funny'].quantile(0.5),
                            'very funny': df['funny'].max()}
        cool_thresholds = {'not cool': df['cool'].quantile(0.25),
                           'neutral': df['cool'].quantile(0.5),
                           'very cool': df['cool'].max()}

        df['stars_transformed'] = df['stars'].apply(stars_to_sentiment)
        df['useful_transformed'] = df['useful'].apply(lambda x: categorize_value(x, useful_thresholds))
        df['funny_transformed'] = df['funny'].apply(lambda x: categorize_value(x, funny_thresholds))
        df['cool_transformed'] = df['cool'].apply(lambda x: categorize_value(x, cool_thresholds))

        df.drop(columns=['stars', 'useful', 'funny', 'cool'], inplace=True)

        # Feature Extraction
        X = df['text']
        y = df[['stars_transformed', 'useful_transformed', 'funny_transformed', 'cool_transformed']]

        # Split dataset (train 70%, val 15%, test 15%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        # Vectorize text data (N-grams & Bag of Words)
        ngram_range = (1, 2)
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_validation_vectorized = vectorizer.transform(X_val)
        X_test_vectorized = vectorizer.transform(X_test)

        # Save and pickle vectorizer
        pickle_model(vectorizer, save_path=JOHNNY_EXPERIMENT_THREE_PKL_VECTORIZE_DIR)

        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced']
        }

        # Train classifiers and perform hyperparameter tuning per target variable
        best_estimators = {}
        for target in y_train.columns:
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(X_train_vectorized, y_train[target])

            # Perform GridSearchCV for hyperparameter tuning using the validation set
            grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_validation_vectorized, y_val[target])

            # Print best parameter setting for each target variable
            print("[+] Best Parameters for MaxEnt when Predicting Target Variable ({}): {}"
                  .format(target, grid_search.best_params_))

            # Retrieve the best estimator and store it
            best_estimators[target] = grid_search.best_estimator_

        # Evaluate the hyper-tuned classifiers (best estimators) on the test set
        for target, best_estimator in best_estimators.items():
            y_pred_test = best_estimator.predict(X_test_vectorized)

            # Save and pickle the classifier
            pickle_model(best_estimator, save_path=JOHNNY_EXPERIMENT_THREE_PKL_DIR, target=target)

            print(BORDER)
            print(f"Classification Report for Target Variable ('{target}') on Test Set: ")
            print(classification_report(y_test[target], y_pred_test, zero_division=1))
            print(BORDER)

    @staticmethod
    def perform_experiment_three_inference(file_path: str):
        """
        Performs experiment 3 but in inference mode
        (using pickled models).

        @param file_path:
            A string representing the file path
            of test JSON file

        @return: None
        """

        def stars_to_sentiment(stars: int):
            if stars <= 2:
                return 'negative'
            elif stars == 3:
                return 'neutral'
            else:
                return 'positive'

        def categorize_value(value: int, thresholds: dict):
            for category, threshold in thresholds.items():
                if value <= threshold:
                    return category
            return None

        print("[+] Now performing Johnny's experiment 3 in inference mode...")

        # Load JSON test dataset
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
            test_df = pd.DataFrame(data)

        # Categorize 'useful', 'funny', and 'cool' columns
        useful_thresholds = {'not useful': test_df['useful'].quantile(0.25),
                             'neutral': test_df['useful'].quantile(0.5),
                             'very useful': test_df['useful'].max()}
        funny_thresholds = {'not funny': test_df['funny'].quantile(0.25),
                            'neutral': test_df['funny'].quantile(0.5),
                            'very funny': test_df['funny'].max()}
        cool_thresholds = {'not cool': test_df['cool'].quantile(0.25),
                           'neutral': test_df['cool'].quantile(0.5),
                           'very cool': test_df['cool'].max()}

        test_df['stars_transformed'] = test_df['stars'].apply(stars_to_sentiment)
        test_df['useful_transformed'] = test_df['useful'].apply(lambda x: categorize_value(x, useful_thresholds))
        test_df['funny_transformed'] = test_df['funny'].apply(lambda x: categorize_value(x, funny_thresholds))
        test_df['cool_transformed'] = test_df['cool'].apply(lambda x: categorize_value(x, cool_thresholds))

        test_df.drop(columns=['stars', 'useful', 'funny', 'cool'], inplace=True)

        # Feature Extraction
        X_test = test_df['text']
        y_test = test_df[['stars_transformed', 'useful_transformed', 'funny_transformed', 'cool_transformed']]

        # Load vectorizer model and vectorize X_test
        with open(JOHNNY_EXPERIMENT_THREE_PKL_VECTORIZE_DIR, 'rb') as f:
            vectorizer = pickle.load(f)
            X_test_vectorized = vectorizer.transform(X_test)

        # Evaluate each classifier
        for target in y_test.columns:
            with open(JOHNNY_EXPERIMENT_THREE_PKL_DIR.format(target), 'rb') as f:
                classifier = pickle.load(f)
                y_pred_test = classifier.predict(X_test_vectorized)

                print(BORDER)
                print(f"A Classification Report for Categorized Target Variable ('{target}') on Test Set: ")
                print(classification_report(test_df[target], y_pred_test, zero_division=1))
                print(BORDER)
