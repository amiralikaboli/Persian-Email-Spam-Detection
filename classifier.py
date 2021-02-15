import json
import math
import random
import statistics

import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from vectorizer import TfidfVectorizer


class KNN:
    def __init__(self, top_k, num_features=None):
        self.is_fitted = False

        self.top_k = top_k
        self.num_features = num_features

        self.vectorizer = TfidfVectorizer()
        if self.num_features is not None:
            self.feature_selector = SelectKBest(chi2, k=num_features)

    def fit(self, training_texts: list, targets: list):
        self.training_tfidf_vectors = self.vectorizer.fit_transform(training_texts)
        self.targets = targets

        if self.num_features is not None:
            self.training_tfidf_vectors = self.feature_selector.fit_transform(self.training_tfidf_vectors, self.targets)

        self.is_fitted = True

    def predict(self, prediction_texts: list):
        if not self.is_fitted:
            raise Exception('classifier must be fit!')

        prediction_tfidf_vectors = self.vectorizer.transform(prediction_texts)
        if self.num_features is not None:
            prediction_tfidf_vectors = self.feature_selector.transform(prediction_tfidf_vectors)

        similarities = cosine_similarity(prediction_tfidf_vectors, self.training_tfidf_vectors)

        predicted_labels = []
        for ind in range(len(similarities)):
            nearest_indexes = np.argsort(similarities[ind])[::-1][:self.top_k]
            nearest_targets = [self.targets[near_index] for near_index in nearest_indexes]
            predicted_labels.append(statistics.mode(nearest_targets))

        return predicted_labels


class NaiveBayes:
    def __init__(self):
        self.is_fitted = False

    def fit(self, training_texts: list, targets: list):
        self.classes = np.unique(targets)

        self.class_probs = {class_name: targets.count(class_name) / len(targets) for class_name in self.classes}

        self.word_in_classes_frequency = {class_name: {} for class_name in self.classes}
        vocabulary = set()
        for text, target in zip(training_texts, targets):
            tokens = text.split()
            vocabulary = vocabulary.union(tokens)

            for token in tokens:
                if token not in self.word_in_classes_frequency[target]:
                    self.word_in_classes_frequency[target][token] = 0
                self.word_in_classes_frequency[target][token] += 1

        self.vocabulary_length = len(vocabulary)

        self.is_fitted = True

    def predict(self, prediction_texts: list):
        if not self.is_fitted:
            raise Exception('classifier must be fit!')

        predicted_labels = []
        classes_prob_logs = {class_name: 0 for class_name in self.classes}
        for text in prediction_texts:
            tokens = text.split()
            for class_name in self.classes:
                classes_prob_logs[class_name] = math.log10(self.class_probs[class_name])
                for token in tokens:
                    token_freq = 0
                    if token in self.word_in_classes_frequency[class_name]:
                        token_freq = self.word_in_classes_frequency[class_name][token]

                    classes_prob_logs[class_name] += math.log10(
                        (token_freq + 1) /
                        (sum(self.word_in_classes_frequency[class_name].values()) + self.vocabulary_length)
                    )

            predicted_labels.append(max(classes_prob_logs, key=classes_prob_logs.get))

        return predicted_labels


if __name__ == '__main__':
    with open('data/train_data.json', 'r') as json_file:
        train_data = json.load(json_file)
        random.shuffle(train_data)
        train_texts = [data_point[0] for data_point in train_data]
        train_targets = [data_point[1] for data_point in train_data]
    with open('data/test_data.json', 'r') as json_file:
        test_data = json.load(json_file)
        random.shuffle(test_data)
        test_texts = [data_point[0] for data_point in test_data]
        test_targets = [data_point[1] for data_point in test_data]

    # classifier = KNN(5)
    classifier = KNN(5, 500)
    # classifier = NaiveBayes()

    classifier.fit(train_texts, train_targets)
    predicted_labels = classifier.predict(test_texts)

    report = classification_report(test_targets, predicted_labels, digits=4)
    print(report)

    confusion_mat = confusion_matrix(test_targets, predicted_labels)
    print('confusion matrix:', confusion_mat, sep='\n')
