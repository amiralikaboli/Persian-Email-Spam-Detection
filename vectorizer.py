import math

import numpy as np
from scipy.sparse import csr_matrix


class TfidfVectorizer:
    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary

        self.is_fitted = False

    def fit(self, raw_documents):
        fitted_vocabulary = set()

        self.term_frequency = {}
        self.doc_frequency = {}
        for document in raw_documents:
            tokens = document.split()
            unique_tokens = np.unique(tokens)
            for unique_token in unique_tokens:
                if unique_token not in self.doc_frequency:
                    self.doc_frequency[unique_token] = 0
                self.doc_frequency[unique_token] += 1

            fitted_vocabulary = fitted_vocabulary.union(unique_tokens)

        if self.vocabulary is None:
            self.vocabulary = sorted(list(fitted_vocabulary))
        self.vocabulary2index = {word: ind for ind, word in enumerate(self.vocabulary)}

        self.is_fitted = True

    def transform(self, raw_documents):
        row = []
        col = []
        data = []

        num_documents = len(raw_documents)
        for doc_id, document in enumerate(raw_documents):
            term_frequency = {}

            tokens = document.split()
            for token in tokens:
                if token not in self.vocabulary:
                    continue

                if token not in term_frequency:
                    term_frequency[token] = 0
                term_frequency[token] += 1

            for term in term_frequency.keys():
                row.append(doc_id)
                col.append(self.vocabulary2index[term])

                tf = 1 + math.log10(term_frequency[term])
                idf = math.log10(num_documents / self.doc_frequency[term])
                data.append(tf * idf)

        return csr_matrix((data, (row, col)), shape=(num_documents, len(self.vocabulary)))

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)
