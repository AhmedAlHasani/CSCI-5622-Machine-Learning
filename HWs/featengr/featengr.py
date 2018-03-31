import numpy as np
import pandas as pd
import matplotlib.pylab as plt
#%matplotlib inline 

'''
TD IDF and CountVectorizer
binary = false, which means a binary text model is not set, hence, bag-of-words will be used
min_df = minimum document frequency. If terms did not occur in many documents, ignore that term. 
max_df = opposite to above
strip_accents = ascii / unicode / none
sublinear_tf = sublear tf scaling, replace tf with 1+log(tf)
ngrams = unigram, bigrams or ngrams. I used bigrams
norm = to normalize, default is l2
'''

'''
Function Transformer
validate: checks the array X beforehand.  True or False
accept_sparse: True or False
'''
import nltk
import math
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize 
from nltk import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from nltk.stem.porter import *
from nltk.tag import pos_tag
from sklearn import preprocessing

class TDIDF_Stemming():
	def __init__(self):
		self.stemmer = PorterStemmer()
	
	def stem_tokens(tokens, stemmer):
	    stemmed = []
	    
	    for item in tokens:
	        stemmed.append(stemmer.stem(item))
	    
	    return stemmed

	def __call__(self, examples):
		tokens = word_tokenize(examples)
		tokens = [i for i in tokens if i not in string.punctuation and i != "michael" ]
		stemmed = self.stem_tokens(tokens, self.stemmer)
		return stemmed

#Count the length of a sentence 
class sentence_length_transformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, examples):
		return self

	def transform(self, examples):
		X = np.zeros((len(examples), 1))
		
		for ii, x in enumerate(examples):
			X[ii,:] = np.array([len(x)])

		return csr_matrix(X)

class POS_tokenizer(object):
    def __call__(self, text):
        words = word_tokenize(text)
        words_and_pos_tags = pos_tag(words)
        return [word_and_pos[0] + '=' + word_and_pos[1] for word_and_pos in words_and_pos_tags if word_and_pos[1] != "NN" \
        and word_and_pos[1] != "IN" and word_and_pos[1] != "PRP$" and word_and_pos[1] != "VBN"]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from csv import DictReader, DictWriter
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression 

class FeatEngr:
	def __init__(self):
        self.Y = ['True', 'False']
        self.vectorizer = FeatureUnion([(
            "sentence_tfidfVect", Pipeline([('sentece', FunctionTransformer(lambda x:x[0], validate = False)),
                ('tfid', TfidfVectorizer(ngram_range=(1,2), lowercase=True, stop_words='english'))])),
            ("trope_countVect", Pipeline([('trope', FunctionTransformer(lambda x:x[1], validate = False)), 
                ('countvectorizer', CountVectorizer())]))
            ])
        
        self.data = pd.read_csv("../data/spoilers/train.csv")
	
	def build_train_features(self, examples):
		"""
		Method to take in training text features and do further feature engineering 
		Most of the work in this homework will go here, or in similar functions  
		:param examples: currently just a list of forum posts  
		"""
		
		return self.vectorizer.fit_transform(examples)

	def get_test_features(self, examples):
		"""
		Method to take in test text features and transform the same way as train features 
		:param examples: currently just a list of forum posts  
		"""
		return self.vectorizer.transform(examples)

	def show_top10(self):
		"""
		prints the top 10 features for the positive class and the 
		top 10 features for the negative class. 
		"""
		#feature_names = np.asarray(self.vectorizer.get_feature_names())
		#print(self.feature_names)
		top10 = np.argsort(self.logreg.coef_[0])[-10:]
		bottom10 = np.argsort(self.logreg.coef_[0])[:10]
		print("Pos: %s" % " ".join(feature_names[top10]))
		print("Neg: %s" % " ".join(feature_names[bottom10]))

    def train_model(self, random_state=1234):
        """ 
        Method to read in training data from file, and 
        train Logistic Regression classifier. 
        
        :param random_state: seed for random number generator 
        """
        
        #write data in dictionary, convert to list
        dfTrain = list(DictReader(open("../data/spoilers/train.csv")))

        #grab different information from dictionary above
        self.X_train = self.build_train_features([[x["sentence"] for x in dfTrain], [x["trope"] for x in dfTrain]])
        
        #grab spoilers, convert them to 0's and 1's
        self.y_train = np.array(list(['True', 'False'].index(x["spoiler"]) for x in dfTrain))
        
        k_folds_test = KFold(n_splits=10, shuffle=True)
        accuracy = []
        for train_index, test_index in k_folds_test.split(self.X_train):
            local_x_train, local_x_test = self.X_train[train_index], self.X_train[test_index]
            local_y_train, local_y_test = self.y_train[train_index], self.y_train[test_index]

            self.logreg = LogisticRegression(random_state=1230)
            self.logreg.fit(local_x_train, local_y_train)
            local_y_pred = self.logreg.predict(local_x_test)
            accurate = accuracy_score(local_y_test, local_y_pred)

            accuracy.append(accurate)
            print('Local Accuracy: ', accurate)
        
        print('Avg Accuracy is: ', sum(accuracy) / len(accuracy))

        #train logistic regression model.  !!You MAY NOT CHANGE THIS!! 
        self.logreg = LogisticRegression(random_state=random_state)
        self.logreg.fit(self.X_train, self.y_train)

        scores = cross_val_score(self.logreg, self.X_train, self.y_train, cv =10)
        print(scores)

    def model_predict(self):
        """
        Method to read in test data from file, make predictions
        using trained model, and dump results to file 
        """
        # read in test data 
        dfTest = list(DictReader(open("../data/spoilers/test.csv")))
        
        # featurize test data 
        self.X_test = self.get_test_features([[x["sentence"] for x in dfTest], [x["trope"] for x in dfTest]])
        
        # make predictions on test data 
        pred = self.logreg.predict(self.X_test)

        #increment id as each line is written
        id_csv = 0
        
        # dump predictions to file for submission to Kaggle  
        with open("prediction.csv", "w") as output:
            wr = DictWriter(output, fieldnames=["Id", "spoiler"], lineterminator = '\n')
            wr.writeheader()

            for p in pred:
                d = {"Id": id_csv, "spoiler": self.Y[p]}
                wr.writerow(d)
                id_csv+=1

	def computeLength(self):

		count_true = 0.0
		count_false = 0.0
		total_true_length = 0.0
		total_false_length = 0.0
		for index, row  in self.data.iterrows():
			if row["spoiler"] == True:
				count_true += 1.0
				total_true_length += len(row["sentence"])
			else:
				count_false += 1.0
				total_false_length += len(row["sentence"])

		print("Avg Length For Spoilers: ", end="")
		print(total_true_length/count_true)
		print("Avg Length For Non-Spoilers: ", end="")
		print(total_false_length/count_false)
		print("Total No. of Spoilers: " + str(count_true))
		print("Total No. of Non-Spoilers: " + str(count_false))

def main():
	# Instantiate the FeatEngr clas 
	feat = FeatEngr()
	# Train your Logistic Regression classifier 

	feat.train_model(random_state=1230)

	# Make prediction on test data and produce Kaggle submission file 
	feat.model_predict()

	# Shows the top 10 features for each class 
	#feat.show_top10()

main()