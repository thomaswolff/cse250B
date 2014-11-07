import subprocess
import numpy
from sets import Set

class Dataset:

	def __init__(self, sigma, d):
		self.sigma = sigma
		self.d = d
		self.id = 0
		self.lexicon = {}
		self.stopwords = self.readStopWords()
		self.mapWordToId = {}
		self.mapIdToWord = {}
		self.reviews = self.readReviews()

	def readStopWords(self):
	    lines = self.readFile("../data/stopwords.txt")
	    stopwords = Set([])
	    for line in lines:
	        words = line.split(' ')
	        for word in words:
	            stopwords.add(word)
	    return stopwords

	def readFile(self, filename):
	    with open (filename) as data:
	        lines = data.readlines()
	    return lines

	def readFilenames(self, folder):
		p = subprocess.Popen(["ls " + folder],  stdout=subprocess.PIPE, shell=True)
		(negativeFiles, err) = p.communicate()
		return negativeFiles.strip().split('\n')

	def readReviews(self):
		reviews = []
		files = self.readFilenames("../data/review_polarity/txt_sentoken/neg")
		for filename in files:
			reviews.append(self.readReview("../data/review_polarity/txt_sentoken/neg/" + filename, [0]))
		files = self.readFilenames("../data/review_polarity/txt_sentoken/pos")
		for filename in files:
			reviews.append(self.readReview("../data/review_polarity/txt_sentoken/pos/" + filename, [1]))
		return reviews

	def readReview(self, filename, label):
		lines = self.readFile(filename)
		sentences = []
		for line in lines:
			s = filter(lambda x: x not in {'(', ')', '.', ',', '?', '!', ':', ";", '"', '-', '`'}, line)
			s = filter(lambda x: x not in {''}, s.strip().split(' '))
			if len(s) > 1:
				wordIds = []
				for word in s:
					if  word not in self.mapWordToId:
						self.lexicon[self.id] = numpy.random.normal(0.0, self.sigma, self.d)
						self.mapWordToId[word] = self.id
						self.mapIdToWord[self.id] = word
						wordIds.append(self.id)
						self.id += 1
					else:
						wordIds.append(self.mapWordToId[word])
				if len(wordIds) > 1:
					sentences.append(Sentence(wordIds, label, self.mapIdToWord))
		return Review(sentences, label)


class Sentence:
	def __init__(self, words, label, mapIdToWord):
		self.label = label
		self.words = words
		self.prediced = None
		self.mapIdToWord = mapIdToWord

	def __str__(self):
		return str([self.mapIdToWord[index] for index in self.words])


class Review:
	def __init__(self, sentences, label):
		self.label = label
		self.sentences = sentences
		self.prediced = None


