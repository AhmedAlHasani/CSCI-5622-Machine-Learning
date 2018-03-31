import numpy as np
import matplotlib.pylab as plt
import math
from collections import Counter
#%matplotlib inline

kSEED = 1735
kBIAS = "BIAS_CONSTANT"

np.random.seed(kSEED)

class Example:
    """
    Class to represent a document example
    """
    def __init__(self, label, words, vocab):
        """
        Create a new example

        :param label: The label (0 / 1) of the example, this of this as 'y' #this is positive
        :param words: The words in a list of "word:count" format            #this is negative
        :param vocab: The vocabulary to use as features (list)              #features
        """
        self.nonzero = {}
        self.y = label
        self.x = np.zeros(len(vocab))
        for word, count in [x.split(":") for x in words]: #split words into a word and its count
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count) #vocab.index(word) returns index, hence, self.x[index] is increased by 1 each time
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1 #bias?

def read_dataset(positive, negative, vocab, train_frac=0.9): #is there is a training fraction, it must be SGD?
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_frac: How much of the data should be reserved for test
    """
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x] #what does this mean? if '\t' in x
    
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train_set = []
    test_set = []
    for label, input in [(1, positive), (0, negative)]: #what does this mean? I understand input opens negative, but what about 0?
        #print(str(label) + input)
        for line in open(input): #why open here: because input is a file. Opens positive first, then negative. 1 is +ve, 0 is -ve
            ex = Example(label, line.split(), vocab) #class intializer is called. 
            if np.random.random() <= train_frac:
                train_set.append(ex) #majority will be held here, each train_set has a nonzero dictionary, y label and x NumPy Array
            else:
                test_set.append(ex)

    # Shuffle the data 
    np.random.shuffle(train_set) #important to update coefficents
    np.random.shuffle(test_set)
    return train_set, test_set, vocab

class LogReg:
    def __init__(self, train_set, test_set, lam, eta=0.1):
        """
        Create a logistic regression classifier

        :param train_set: A set of training examples
        :param test_set: A set of test examples 
        :param lam: Regularization parameter
        :param eta: The learning rate to use 
        """
        
        # Store training and test sets 
        self.train_set = train_set
        self.test_set = test_set 
        
        # Initialize vector of weights to zero  
        self.w = np.zeros_like(train_set[0].x)

        # Store regularization parameter and eta function 
        self.lam = lam
        self.eta = eta
        print("eta is : " + str(self.eta))
        print("lam: " + str(self.lam))
        # Create dictionary for lazy-sparse regularization
        self.last_update = dict()

        # Make sure regularization parameter is not negative 
        assert self.lam>= 0, "Regularization parameter must be non-negative"
        
        # Empty lists to store NLL and accuracy on train and test sets 
        self.train_nll = []
        self.test_nll = []
        self.train_acc = []
        self.test_acc = []
        self.iterations = []
        
    def sigmoid(self,score, threshold=20.0):
        """
        Prevent overflow of exp by capping activation at 20.
        You do not need to change this function. 

        :param score: A real valued number to convert into a number between 0 and 1
        """

        # if score > threshold, cap value at score 
        if abs(score) > threshold:
            score = threshold * np.sign(score)

        return 1.0 / (1.0 + np.exp(-score)) 

    def compute_progress(self, examples):
        """
        Given a set of examples, compute the NLL and accuracy
        You shouldn't need to change this function. 

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """
        NLL = 0.0
        num_correct = 0
        for ex in examples:
            # compute prob prediction
            p = self.sigmoid(self.w.dot(ex.x))
            # update negative log likelihood
            NLL = NLL - np.log(p) if ex.y==1 else NLL - np.log(1.0-p) 
            # update number correct 
            num_correct += 1 if np.floor(p+.5)==ex.y else 0
        
        return NLL, float(num_correct) / float(len(examples))
    
    def train(self, num_epochs=1, isVerbose=False, report_step=5):
        """
        Train the logistic regression classifier on the training data 

        :param num_epochs: number of full passes over data to perform 
        :param isVerbose: boolean indicating whether to print progress
        :param report_step: how many iterations between recording progress
        """
        iteration = 0
        # Perform an epoch 
        for pp in range(num_epochs):
            # shuffle the data  
            np.random.shuffle(self.train_set)
            # loop over each training example
            for ex in self.train_set:
                # perform SGD update of weights 
                self.sgd_update(ex, iteration)
                # record progress 
                if iteration % report_step == 1: 
                    train_nll, train_acc = self.compute_progress(self.train_set)
                    test_nll, test_acc = self.compute_progress(self.test_set)
                    self.train_nll.append(train_nll)
                    self.test_nll.append(test_nll)
                    self.train_acc.append(train_acc)
                    self.test_acc.append(test_acc)
                    self.iterations.append(iteration)
                    if isVerbose:
                        print("Update {: 5d}  TrnNLL {: 8.3f}  TstNLL {: 8.3f}  TrnA {:.3f}  TstA {:.3f}"
                             .format(iteration-1, train_nll, test_nll, train_acc, test_acc))
                iteration += 1
        return max(self.test_acc)

    def sgd_update(self, train_example, iteration):
        """
        Compute a stochastic gradient update to improve the NLL 

        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        """
        
        # TODO implement LSR updates of weights
        gradient = (self.sigmoid(np.dot(self.w, train_example.x)) - train_example.y)*train_example.x
        self.w = self.w - gradient*self.eta

        shrinkage = 1 - 2*self.eta*self.lam 
        for x_index, x_elem in enumerate(train_example.x):
            #dont grab the first element, and only regularize non-zero x's
            if x_index != 0 and x_elem > 0.0:
                #initialize feature updated to -1, either it will be changed if x is in last update or it will add to unchaged features
                last_iteration_updated = -1
                if x_index in self.last_update:
                    #Last time feature was updated
                    last_iteration_updated = self.last_update[x_index]
                    #raise Shrinkage Factor to power of no. of iterations since feature was updated
                shrinkage = shrinkage**(iteration - last_iteration_updated)
                #regularize
                self.w[x_index]  = self.w[x_index]*shrinkage
                #store iteration for that feature
                self.last_update[x_index] = iteration

        return self.w


def lam(train_set, test_set):
    lam_values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    lam_accuracies = []
    for x in lam_values:
        model = LogReg(train_set, test_set, lam = x, eta = 0.01)
        lam_accuracies.append(model.train())

    plt.title("Accuracy vs. Lam with eta=0.01")
    plt.plot(lam_values, lam_accuracies, color='g')
    plt.show()
    print(lam_accuracies)
    print("Highest Accuracy from Lam Values of: " + str(lam_values[lam_accuracies.index(max(lam_accuracies))]))
    return lam_values[lam_accuracies.index(max(lam_accuracies))]

def eta(train_set, test_set, max_lam): 
    eta_values = [0.01, 0.1, 0.5, 1.0, 2.0]
    eta_accuracies = []
    fig = plt.figure(figsize=(15, 9))
    for x in eta_values:
        if x == 0.01:
            word_model =  LogReg(train_set, test_set, lam = max_lam, eta = x)
            word_model.train()
        model = LogReg(train_set, test_set, lam = max_lam, eta = x)
        model.train()
        plt.plot(model.iterations, model.test_acc)
    plt.title("Accuracy vs. ETA with lam=0.0")
    plt.legend(['eta = 0.01', 'eta = 0.1', 
        'eta = 0.5', 'eta = 1.0', 'eta =2.0'], loc='lower right')
    plt.show()
    return word_model

def top_words1(train_set, test_set):
    model = LogReg(train_set, test_set, lam = 0.0, eta = 0.01)
    model.train()
    k = model.w
    top_ten = k.argsort()[-10:][::-1] #grab indices of highest ten elements
    top_ten_words = []
    top_ten_weights = []
    for element in top_ten:
        top_ten_weights.append(k[element])
        top_ten_words.append(vocab[element])
    print("Best Motorcycles Predictors")
    print(top_ten_words)
    print(top_ten_weights)

def top_words0(train_set, test_set):
    model = LogReg(train_set, test_set, lam = 0.0, eta = 0.01)
    model.train()
    k = model.w
    least_ten = k.argsort()[:10] #grab indices of least ten elements
    least_ten_words = []
    least_ten_weights = []
    for element in least_ten:
        least_ten_weights.append(k[element])
        least_ten_words.append(vocab[element])
    print("Best Automobile Predictors")
    print(least_ten_words)
    print(least_ten_weights)

def worst_predictors_1(train_set, test_set):
    model = LogReg(train_set, test_set, lam = 0.0, eta = 0.01)
    model.train()
    w = model.w
    w[w < 0] = math.inf #change weights that are negative to +ve infinity, for the line below to work
    least_ten = w.argsort()[:10] #grab the index of the least 10 words
    top_ten_words = []
    top_ten_weights = []
    for elemnet in least_ten:
        top_ten_weights.append(w[elemnet])
        top_ten_words.append(vocab[elemnet])
    print("Worst Motorcycles Predictors")
    print(top_ten_words)
    print(top_ten_weights)

def worst_predictors_0(train_set, test_set):
    model = LogReg(train_set, test_set, lam = 0.0, eta = 0.01)
    model.train()
    w = model.w
    w[w > 0] = -math.inf #change weights that are negative to -ve infinity, for the line below to work
    top_ten = w.argsort()[-10:][::-1] #grab the index of the highest 10 values
    top_ten_words = []
    top_ten_weights = []
    for elemnet in top_ten:
        top_ten_weights.append(w[elemnet])
        top_ten_words.append(vocab[elemnet])
    print("Worst Automobile Predictors")
    print(top_ten_words)
    print(top_ten_weights)

