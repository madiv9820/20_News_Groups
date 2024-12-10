import numpy as np

class MultinomialNB:
    def __init__(self) -> None:
        self.__count_Dict = {}  # Dictionary to hold counts and statistics for each class
        self.__Inputs = None  # Input features
        self.__Outputs = None  # Output labels
        self.__log_priors = {}  # Log priors for each class

    # String representation of the class
    def __str__(self) -> str:
        return 'naive_bayes.MultinomialNB()'

    def __create_Dictionary(self) -> None:
        """
        Build the count dictionary for each class and each feature.
        This dictionary will store the occurrence of each feature value for every class.
        """
        for output in np.unique(self.__Outputs):
            self.__count_Dict[output] = {}
            # Store the count of occurrences of the class (for prior computation)
            self.__count_Dict[output]['Count'] = np.sum(self.__Outputs == output)

            # Vectorized counting of unique feature values for each feature and class
            feature_counts = [
                np.unique(self.__Inputs[self.__Outputs == output, i], return_counts=True)
                for i in range(self.__Inputs.shape[1])
            ]

            # Store the counts for each unique feature value
            for feature_No, (unique_features, counts) in enumerate(feature_counts):
                self.__count_Dict[output][feature_No] = dict(zip(unique_features, counts))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the training data.
        This builds the necessary count dictionary and computes class prior probabilities.
        """
        self.__Inputs = x
        self.__Outputs = y
        self.__create_Dictionary()  # Build the count dictionary

        # Precompute log priors for each class (log of class frequencies)
        total_samples = len(self.__Outputs)
        self.__log_priors = {
            output: np.log(self.__count_Dict[output]['Count']) - np.log(total_samples)
            for output in self.__count_Dict
        }

    def __calculate_log_likelihood(self, sample, output):
        """
        Calculate the log-likelihood of a sample given a class.
        Uses the counts of feature values in the class with Laplace correction.
        """
        log_likelihood = 0.0
        count_dict = self.__count_Dict[output]  # Cache the dictionary for the class
        
        for feature_No, feature_value in enumerate(sample):
            if feature_No in count_dict:
                feature_dict = count_dict[feature_No]
                # Laplace correction: add 1 to the count, and adjust the total count
                count = feature_dict.get(feature_value, 0) + 1  # Add 1 for unseen features
                total_count = count_dict['Count'] + len(feature_dict)  # + unique feature values
                log_likelihood += np.log(count) - np.log(total_count)  # Log probability
        
        return log_likelihood

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input samples.
        For each sample, calculate the log posterior probabilities and choose the class with the maximum probability.
        """
        predictions = []
        
        # Iterate through each sample to predict the class
        for sample in x:
            class_log_probabilities = {}  # Store log probabilities for each class
            
            # Compute the total log probability (prior + likelihood) for each class
            for output in self.__count_Dict:
                likelihood = self.__calculate_log_likelihood(sample, output)  # Log likelihood
                class_log_probabilities[output] = self.__log_priors[output] + likelihood  # Total log probability
            
            # Choose the class with the highest log probability
            predicted_class = max(class_log_probabilities, key=class_log_probabilities.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)

    def score(self, x: np.ndarray, y: np.ndarray) -> np.float64:
        """
        Calculate the accuracy of the model on test data.
        Compares predicted labels with the actual labels.
        """
        outputs = self.predict(x)  # Get predictions
        return np.sum(outputs == y) / y.shape[0]  # Calculate accuracy as the proportion of correct predictions