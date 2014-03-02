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

	def getProbability(self, tettas):
		xSum = 0.0
		tettaSum = 0.0
		for key, value in self.count_vector.iteritems():
			print "value: " + str(value)
			print "log x: " + str(math.log(math.factorial(value)))
			xSum += math.log(math.factorial(value))
			tettaSum += value * math.log(tettas[key])

		result = math.log(math.factorial(self.length))
		print "1: " + str(result)
		result -= xSum
		print "2: " + str(xSum)
		result += tettaSum
		print "3: " + str(tettaSum)
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
	index = 1
	for line in lines:
		arr = line.rstrip().split(' ')
		vocabulary[index] = arr[1]
		index += 1
	return vocabulary

def read400Terms():
	lines = readFile("data/classic400/wordlist.txt")
	vocabulary = {}
	index = 1
	for line in lines:
		line = line.rstrip()
		vocabulary[index] = line
		index += 1
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


def test():
	# docs : documents which consists of word array
	# K : number of topics
	# V : vocaburary size

	z_m_n = [] # topics of words of documents
	n_m_z = numpy.zeros((len(self.docs), K)) + alpha     # word count of each document and topic
	n_z_t = numpy.zeros((K, V)) + beta # word count of each topic and vocabulary
	n_z = numpy.zeros(K) + V * beta    # word count of each topic

	for m, doc in enumerate(docs):
	    z_n = []
	    for t in doc:
	        # draw from the posterior
	        p_z = n_z_t[:, t] * n_m_z[m] / n_z
	        z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

	        z_n.append(z)
	        n_m_z[m, z] += 1
	        n_z_t[z, t] += 1
	        n_z[z] += 1
	    z_m_n.append(numpy.array(z_n))

def gibbs(documents, k, v, alpha, beta):
	zs = initializeZs(k)


def initializeZs(zs, k):
	z = []
	for z in zs:
		z.append(random.randint(0, k))
	return z

def main():
	c = 0.1
	documents = readFileNames()
	vocabulary = readTerms()
	readTermMatrix(documents)
	sumOfSizes = 0
	tettas = [0 for i in range(len(vocabulary))]
	calculateTettas(tettas, vocabulary, c)
	print str(tettas)
	print documents[95].getProbability(tettas)

	sum = 0.0
	for tet in tettas:
		sum+= tet
	print sum

main()



