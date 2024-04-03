import json
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Constants
BORDER = "================================================================"


class MultinomialNB:
    """
    Naive Bayes classifier for multinomial models.

    Attributes:
        alpha: A smoothing parameter to handle unseen words
        class_prior_probabilities: Prior probabilities of each class.
        feature_likelihood_probabilities: Likelihood probabilities of each word given each class.
    """
    def __init__(self, alpha=1.0):
        """
        A constructor for Multinomial NB.
        @param alpha:
            A smoothing parameter to handle unseen words
        """
        self.alpha = alpha
        self.class_prior_probabilities = None
        self.feature_likelihood_probabilities = None

    def fit(self, X, y):
        """
        Train (fit) the model according to input features
        and labels.

        @param X:
            The training input samples (Independent variables)

        @param y:
            The input target variables we are
            trying to predict (Dependent variables)

        @return: None
        """
        num_samples, num_features = X.shape
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)

        # Compute probabilities of each class
        self.class_prior_probabilities = np.zeros(num_classes)
        for i, c in enumerate(self.classes_):
            self.class_prior_probabilities[i] = np.sum(y == c) / num_samples

        # Count occurrences of each word for each class
        word_counts = np.zeros((num_classes, num_features))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            word_counts[i, :] = np.sum(X_c, axis=0)

        # Compute probabilities of each word given a class with smoothing
        smoothed_word_counts = word_counts + self.alpha
        smoothed_class_totals = np.sum(smoothed_word_counts, axis=1).reshape(-1, 1)
        self.feature_likelihood_probabilities = smoothed_word_counts / smoothed_class_totals

    def __predict_log_probabilities(self, X):
        """
        Predict log probability estimates for the input samples.

        @param X:
            The input test data used to predict y

        @return: log_probs
            The log probability estimates for each sample
            in X for each class.
        """
        # Compute log probability estimates for the test data X
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)

        num_classes = len(self.classes_)
        log_probs = np.zeros((X.shape[0], num_classes))

        for i, c in enumerate(self.classes_):
            # Log of probability of the class
            log_probs[:, i] = np.log(self.class_prior_probabilities[i])

            # Log of conditional probabilities of words given a class
            log_probs[:, i] += X.dot(np.log(self.feature_likelihood_probabilities[i, :]).T)

        return log_probs

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        @param X:
            The input test data used to predict y

        @return: Predicted class labels for each sample in X.
        """
        # Predict class labels for test data X
        log_probs = self.__predict_log_probabilities(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        @param X:
            The test input samples

        @param y:
            The true labels for X

        @return: Mean accuracy of self.predict(X) with respect to y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return {'alpha': self.alpha}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    @staticmethod
    def perform_experiment_one_training(file_path: str):
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

        # Pickle the vectorizer
        with open("pickled_models/naive_bayes/vectorizer.pkl", 'wb') as f:
            pickle.dump(vectorizer, f)

        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        }

        # Train classifiers and perform hyperparameter tuning per target variable
        best_estimators = {}
        for target in y_train.columns:
            classifier = MultinomialNB()
            classifier.fit(X_train_vectorized, y_train[target])

            # Perform GridSearchCV for hyperparameter tuning using the validation set
            grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
            grid_search.fit(X_validation_vectorized, y_validation[target])

            # Print best parameter setting for each target variable
            print("[+] Best Parameters for target variable ({}): {}".format(target, grid_search.best_params_))

            # Pickle the classifier
            with open(f"pickled_models/naive_bayes/multinomialNB_{target}.pkl", 'wb') as f:
                pickle.dump(classifier, f)

            # Retrieve the best estimator and store it
            best_estimators[target] = grid_search.best_estimator_

        # Evaluate the hyper-tuned classifiers (best estimators) on the test set
        for target, best_estimator in best_estimators.items():
            y_pred_test = best_estimator.predict(X_test_vectorized)
            print(BORDER)
            print(f"Classification Report for Target Variable ('{target}') on Test Set: ")
            print(classification_report(y_test[target], y_pred_test, zero_division=1))
            print(BORDER)

    # TODO: Experiment 2: Try unigram (1, 1) and compare results (explain why better or worse?)

    # TODO: Experiment 3: Try a different vectorization technique (maybe one with semantics or meaning)
