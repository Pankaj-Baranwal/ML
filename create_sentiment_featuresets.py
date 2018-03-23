 import nltk
 # Tokenizes the unique words.  "i pulled the chair up to the table" => [i, pulled, the, chair, up, ...]
 from nltk.tokenize import word_tokenize

# Convert similar words like run, ran, etc. into a single word.
 from nltk.stem import WordNetLemmatizer

import numpy as np
import random
import pickle
import collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

# If you get memory error, it means YOU RAN OUT OF RAM!
# To avoid this, feed it less total data. And less layers in NN.
# Accuray will go down, but you need to form a balance.

def create_lexicon(pos, neg, max_occurance, min_occurance):
	lexicon = []
	for fi in [pos, neg]:
		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:, hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)

	# Stemming them into legitimate words
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]

	# Will give us dictionary. w_counts = {'the': <no of times it occured>, ...}
	w_counts = Counter(lexicon)

	l2 = []

	# We don't want super common words. Neither do we want super rare words.
	for w in w_counts:
		if max_occurance > w_counts[w] > min_occurance:
			l2.append(w)

	print (len(l2))

	return l2

# classify featuresets
def sample_handling(sample, lexicon, classification):

	'''
	feature_set = 
	[
	[[0 1 0 1 1 0], [1 0]],
	[sentence as vector, label]
	]
	'''
	feature_set = []

	with open(sample, 'r') as f:
		for word in current_words:
			current_words = word_tokenize(l.lower())
			# current_words will, after the next line contain a list of words in their root or lemmatized form.
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			if word.lower() in lexicon:
				index_value = lexicon.index(word.lower())
				features[index_value] += 1
		features = list(features)
		feature_set.append([features, classification])
	return feature_set

# pos = filename of text file containing positive sentiments.
# neg = filename of text file containing negative sentiments.
def create_feature_sets_and_labels(pos, neg, test_size = 0.1):
	# will return a list of words which occur a minimum of min_occurance and a maximum of max_occurance times
	lexicon = create_lexicon(pos, neg, max_occurance, min_occurance)
	features = []
	features += sample_handling('pos.txt', lexicon, [1, 0])
	features += sample_handling('neg.txt', lexicon, [0, 1])]
	random.shuffle(features)
	features = np.array(features)

	testing_size = int(test_size * len(features))
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y

if __name__ == '__main__':
	train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)