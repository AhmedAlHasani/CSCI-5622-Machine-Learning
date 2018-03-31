import unittest
import numpy as np
from sklearn.neighbors import BallTree
import pickle, gzip
import random

class Numbers:
    """
    Class to store MNIST data
    """
    def __init__(self, location):

        # load data from file 
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        # store for use later  
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

class Knearest:
    """
    kNN classifier
    """
    def __init__(self, X, y, k=5):
        """
        Creates a kNN instance
        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        self._kdtree = BallTree(X)
        self._y = y
        self._k = k
        self._counts = self.label_counts()
        
    def label_counts(self): 
        """
        Given the training labels, return a dictionary d where d[y] is  
        the number of times that label y appears in the training set. 
        """
        dictionary = { }
        for labels in self._y:
            if labels not in dictionary:
                dictionary[labels] = 0
            dictionary[labels] += 1

        return dictionary

    def majority(self, neighbor_indices):
        """
        Given the indices of training examples, return the majority label. Break ties 
        by choosing the tied label that appears most often in the training data. 

        :param neighbor_indices: The indices of the k nearest neighbors
        """
        assert len(neighbor_indices) == self._k, "Did not get k neighbor indices"

        neighbor_labels = [self._y[i] for i in neighbor_indices] #given indices, grab the corresponding labels from self._y
        
        labels_frequency = { }
        for labels in neighbor_labels:
            if labels not in labels_frequency:
                labels_frequency[labels] = 0
            labels_frequency[labels] += 1
            
        maximum = max(labels_frequency, key = labels_frequency.get)
        maximum_label = 0

        if labels_frequency[maximum] == 1:
            for x in labels_frequency:
                if self._counts[x] > maximum_label:
                    maximum_label = self._counts[x]
                    maximum = x

        return maximum
    
    def classify(self, example):
        """
        Given an example, return the predicted label. 

        :param example: A representation of an example in the same
        format as a row of the training data
        """
        dist, ind = self._kdtree.query(np.array(example).reshape(1, -1), k=self._k)
        return self.majority(ind[0])

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a 2-dimensional
        numpy array of ints, C, where C[ii,jj] is the number of times an 
        example with true label ii was labeled as jj.

        :param test_x: test data 
        :param test_y: true test labels 
        """
        C = np.zeros((10,10), dtype=int)

        for xx, yy in zip(test_x, test_y):
            jj = self.classify(xx) #return the predicted label
            C[yy][jj] += 1 #increase by 1 where true label and predicted label intersect

        #print(C)
        return C 
            
    @staticmethod
    def accuracy(C):
        """
        Given a confusion matrix C, compute the accuracy of the underlying classifier.
        
        :param C: a confusion matrix 
        """
        return np.sum(C.diagonal()) / C.sum()

class TestKnn(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[2, 0], [4, 1], [6, 0], [1, 4], [2, 4], [2, 5], [4, 4], [0, 2], [3, 2], [4, 2], [5, 2], [5, 5]])
        self.y = np.array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1])
        self.knn = {}
        for ii in [1, 2, 3]:
            self.knn[ii] = Knearest(self.x, self.y, ii)

        self.queries = np.array([[1, 5], [0, 3], [6, 4]])
        
    def test0(self):
        """
        Test the label counter 
        """
        self.assertEqual(self.knn[1]._counts[-1], 5)
        self.assertEqual(self.knn[1]._counts[1], 7)

    def test1(self):
        """
        Test 1NN
        """
        self.assertEqual(self.knn[1].classify(self.queries[0]),  1)
        self.assertEqual(self.knn[1].classify(self.queries[1]), -1)
        self.assertEqual(self.knn[1].classify(self.queries[2]), -1)

    def test2(self):
        """
        Test 2NN
        """
        self.assertEqual(self.knn[2].classify(self.queries[0]),  1)
        self.assertEqual(self.knn[2].classify(self.queries[1]),  1)
        self.assertEqual(self.knn[2].classify(self.queries[2]),  1)

    def test3(self):
        """
        Test 3NN
        """
        self.assertEqual(self.knn[3].classify(self.queries[0]),  1)
        self.assertEqual(self.knn[3].classify(self.queries[1]),  1)
        self.assertEqual(self.knn[3].classify(self.queries[2]), -1)
        
tests = TestKnn()
tests_to_run = unittest.TestLoader().loadTestsFromModule(tests)
unittest.TextTestRunner().run(tests_to_run)