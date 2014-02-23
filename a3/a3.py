import string, re, timeit, math

NUMBER_OF_DOCUMENTS = 0
NUMBER_OF_TERMS = 0
NUMBER_OF_NON_ZERO_VALUES = 0


# C is a pseudocount. Should have 0<c<=1 and c<T/M
# Tettas is of length M and can be viewed as probability for word j
# T = Sum of sizes of all training data
# V = 
# M = size of vocabulary
# K = number of topics
# n = length of documents in words

class Document(object):

	count_vector = {}
	category = 0
	
	alphas = []
	betas = []

	words = []
	filename = ""
	length = 0

	def __init__(self, filename, category):
		self.category = category
		self.filename = filename
		self.count_vector = {}

	def getProbability():
		denumerator = 0.0
		tettaProduct = 0.0
		for key, value in count_vector.iteritems():
			denumerator *= math.factorial(value)
			tettaProduct *= tettas[key]**value

		result = float(math.factorial(length)) / (denumerator)
		result *= tettaProduct
		return result




		return 0
	def addCount(self, index, count):
		self.count_vector[index] = count
		self.length += count

	def __str__(self):
		print "Document: " + self.filename + "\n"
		print "Category: " + self.category + "\n"
		print "Dict: " + str(self.count_vector)

def findSumOfAllSizes(vocabulary):
	sum = 0
	for key, value in vocabulary.iteritems():
		sum += value[1]
	return sum

def calculateTettas(tettas, vocabulary, c):
	t = findSumOfAllSizes(vocabulary)
	tMarked = (len(vocabulary) * c) + t
	for i in range(len(tettas)):
		allApperances = vocabulary[i+1][1]
		tettas[i] = (1/tMarked) * (c + allApperances)






def readFile(filename):
	with open ("./data/" + filename) as data:
		lines = data.readlines()
	return lines

def readWords(filename):
	lines = readFile(filename)
	words = []
	exclude = set(string.punctuation)
	table = string.maketrans("","")
	for line in lines:
		line = line.lower()
		line = line.rstrip()
		line = line.translate(table, string.punctuation)
		wordsInLine = line.split(' ')
		for word in wordsInLine:
			words.append(word)
	return words

def readTerms():
	lines = readFile("terms_detailed.txt")
	vocabulary = {}
	for line in lines:
		arr = line.rstrip().split(' ')
		vocabulary[int(arr[0])] = (arr[1], int(arr[2]))
	return vocabulary


def readTermMatrix(documents):
	lines = readFile("docbyterm.txt")
	if len(lines) > 0:
		first_line = lines[0].rstrip().split(' ')
		NUMBER_OF_DOCUMENTS = first_line[0]
		NUMBER_OF_TERMS = first_line[1]
		NUMBER_OF_NON_ZERO_VALUES = first_line[2]
	for i in range(1, len(lines)):
		arr = lines[i].rstrip().split(' ')
		if len(arr) > 2:
			doc = documents[int(arr[0]) - 1]
			doc.addCount(int(arr[1]), int(arr[2]))
	print documents[0].count_vector
	print documents[0].filename
	print documents[0].category


def readFileNames():
	raw_lines = readFile("documents.txt")
	documents = []
	for line in raw_lines:
		temp = line.rstrip().split(' ')
		filename = ""
		category = 0
		if (len(temp) > 1):
			filename = temp[0]
			category = temp[1]
		words = readWords("classic/" + filename)
		documents.append(Document(filename, category))
	print documents[0].count_vector
	return documents

def main():
	c = 0.01
	documents = readFileNames()
	vocabulary = readTerms()
	readTermMatrix(documents)
	sumOfSizes = 0
	tettas = [0 for i in range(len(vocabulary))]
	calculateTettas(tettas, vocabulary, c)
	print str(tettas)

main()



