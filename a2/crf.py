import random
import time

START = 0
SPACE = 1
PERIOD = 2
COMMA = 3
QUESTION_MARK = 4
EXCAMLATION_POINT = 5
COLON = 6
STOP = 7
ounter = 0
minint = -100000

question_words = ["how", "where", "who", "why", "when", "what", "can"]
conjunction_words = ["and", "or", "but", "for", "as", "because", "nor", "yet", "so"]
indictment_words = ["hi", "hello", "hey"]
tag_set = [START, SPACE, PERIOD, COMMA, QUESTION_MARK, EXCAMLATION_POINT, COLON, STOP]

class Sentence(object):
	x = []
	y = []
	predicted_y = []

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.predicted_y = []


class Template1D(object):
		x_index = -1
		use_y_i = True
		premise = lambda x: True
		dictionary = {}


	def __init__(self, x_index, use_y_i, set_of_b, premise):
		self.x_index = x_index
		self.use_y_i = use_y_i
		self.premise = premise
		for b in set_of_b:
			dictionary[b] = 0.0

	def __call__(self, y_prev, y, x, i):
		word = ""
		if self.x_index is (-1) word = x[i] else word = x[x_index]
		if self.premise(word):
			if self.use_y_i return self.dictionary[y] else self.dictionary[y_prev]
		else:
			return 0.0

class Template2D(object):
		x_index = -1
		use_y_i = True
		premise = lambda x: True
		dictionary = {}

		def __init__(self, x_index, use_y_i, set_of_a, set_of_b, premise):
			self.x_index = x_index
			self.use_y_i = use_y_i
			self.premise = premise
			for a in set_of_a:
				for b in set_of_b:
					dictionary[str(a)+str(b)] = 0.0

		def __call__(self, y_prev, y, x, i):
			word = ""
			if self.x_index is (-1) word = x[i] else word = x[x_index]
			if self.premise(word):
				if self.use_y_i return self.dictionary[str(word)+str(y)] else self.dictionary[str(word)+str(y_prev)]
			else:
				return 0.0

class FeatureFunction(object):
	a_fun = None
	b_fun = None
	
	def __init__(self, a_fun, b_fun):
		self.a_fun = a_fun
		self.b_fun = b_fun
	
	def __call__(self, words, tags):
		sum = 0.0
		for i in range(len(words)):
			y = tags[i]
			y_prev = START
			if(i is not 0): y_prev = tags[i-1]
			a_result = self.a_fun(words, i)
			if(a_result is not 0):
				sum += a_result * self.b_fun(y_prev, y)
		return sum
		
	def evaluateLowLevelFeatureFunction(self, y_prev, y, words, i):
		a_result = self.a_fun(words, i)
		if(a_result is 0): return 0.0
		else: return a_result * self.b_fun(y_prev, y)
	
	
def featureFunction1(y_prev, y, x, i):
	return y is QUESTION_MARK and x[0].lower() in question_words and i == len(x) - 1


def fFsQuestionTemplate(tag):
	def f(y_prev, y, x, i):
		return y is tag and x[0].lower() in question_words and i == len(x) - 1
	return f

def fFsEndsWithTemplate(suffix, tag):
	def f(y_prev, y, x, i):
		return x[i].endswith(suffix)
	return f

def fFsStartWithCapitalLetter(tag):
	def f(y_prev, y, x, i):
		return x[i][0].isupper() and y is tag
	return f

def fFsStartssWithTemplate(suffix, tag):
	def f(y_prev, y, x, i):
		return x[i].startswith(suffix)
	return f

def fFsConjunction(tag):
	def f(y_prev, y, words, i):
		return y is tag and words[i].lower() in conjunctions_words
	return f

def fFsTag(tag):
	def f(y_prev, y, words, i):
		return y is tag
	return f
	
def fFsTagInLastPostion(tag):
	def f(y_prev, y, words, i):
		return y is tag and i is (len(words)-1)
	return f

# A functions
def startsWithCapitalLetter(x, i):
	return x[i][0].isupper()

def wordStartsWithSequence(a):
	def f(x, i):
		return x[i].startswith(a)
	return f

def wordEndsWithSequence(a):
	def f(x, i):
		return x[i].endswith(a)
	return f

def wordIs(a):
	def f(x, i):
		return x[i].lower() == a
	return f

# B functions
def tagIs(b):
	def f(y_prev, y):
		return y == b
	return f

def prevTagIs(b):
	def f(y_prev, y):
		return y_prev == b
	return f

def tagsAre(b):
	def f(y_prev, y):
		return y_prev == b[0] and y == b[1]
	return f
	

def initializeFeatureFunctions():
	t1D = []
	t2D = []
	t1D.append(Template1D(-1, True, tag_set, lambda x: x in question_words))


# def initializeFeatureFunctions():
# 	f = []
# 	a_functions = []
# 	b_functions = []
# 	for a in {"ing", "s"}:
# 		a_functions.append(wordEndsWithSequence(a))
# 	for a in conjunction_words + question_words:
# 		a_functions.append(wordIs(a))
# 	for tag in tag_set:
# 		b_functions.append(tagIs(tag))
# 		b_functions.append(prevTagIs(tag))
# 		for tag_2 in tag_set:
# 			b_functions.append(tagsAre([tag, tag_2]))
# 	for a_func in a_functions:
# 		for b_func in b_functions:
# 			f.append(FeatureFunction(a_func, b_func))
# 	print("Number of feature functions: " + str(len(f)))
# 	return f




	
def featureFunction2(y_prev, y, x, i):
	return y is STOP








# Calculate value of a g function given y_prev and y
# y_prev and y are the two tags
# w is the vector of weights
# f is a vector of J feature functions
# x is the given sentence
# i is the position in the sequence and also the number of the g-function
def g(y_prev, y, x, i):
	# sum = 0.0
	# for (weight, function) in zip(w, f):
	# 	if (weight != 0.0):
	# 		sum += weight*function.evaluateLowLevelFeatureFunction(y_prev, y, x, i)
	# return sum
	sum = 0.0
	for i in range(len(x)):
		for t in ts:
			sum += ts(y_prev, y, x, i)
	return sum
	
# Calculate all g functions for all y_prev and y given a sentence x
# x is the given sentence, where x is an array of words
# w is the vector of weights
# f is the vector of J feature functions
def preprocess(x, w, f, tag_set):
	gs = []
	for i in range(0, len(x)):
		g_i = []
		for y_prev in tag_set:
			g_i_y_prev = []
			for y in tag_set:
				g_i_y_prev.append(g(y_prev, y, w, f, x, i))
			g_i.append(g_i_y_prev)
		gs.append(g_i)
	return gs

# Find the optimal sequence of length k with v
# k is the length of the sequence
# v is the tag for y[k]
# gs are the different enumerated g-functions


# Fill the U matrix by calling U on every tag in the tag set
def fill_U_matrix(gs, tag_set, n):
	dp_table = [[minint for j in range(len(tag_set))] for i in range(n)]

	for i in range(n):
		for v in tag_set:
			max_val = minint
			if i == 0:
				gs[0][START][v]
			else:
				for u in tag_set:
					u_val = dp_table[i - 1][u] + gs[i][u][v]
					if u_val > max_val:
						max_val = u_val
				dp_table[i][v] = max_val
	return dp_table

# Predict the most likely tag sequence y for sentence x
def predict(gs, tag_set, sentence):
	# Initialize n and y
	n = len(sentence.x)
	y = [START for i in range(n)]
	# Fill U matrix
	U = fill_U_matrix(gs, tag_set, n)
	# Find best prediction for last tag in y
	max_val = minint
	max_u = -1
	for u in range(len(U[n-1])):
		if(U[n-1][u] > max_val):
			max_val = U[n-1][u]
			max_u = u
	y[n-1] = max_u
	# Find best prediction for every other tag in y
	for k in range(n-2, -1, -1):
		max_val = minint
		max_u = -1
		for u in range(len(U[k])):
			u_val = U[k][u] + gs[k+1][u][y[k+1]]
			if(u_val > max_val):
				max_val = u_val
				max_u = u
		y[k] = max_u
	return y

def updateWeightsCP(ws, sentence, fs, y_hat):
	# For each weight, do the update rule in Collins perceptron
	for i in range(len(ws)):
		desired = fs[i](sentence.x, sentence.y)
		#print("F of desired: " + str(desired))
		undesired = fs[i](sentence.x, y_hat)
		#print("F of undesired: " + str(undesired))
		ws[i] += (desired - undesired)
		
def collinsPerceptron(fs, tag_set, training_set, validation_set):
	# Initialize all weights to be zero
	ws = [0.0]*len(fs)
	previousCorrectnessRate = 0.0
	# Epoch loop
	while(True):
		startTime = time.gmtime()
		# For each sentence in the training data, update weights
		for i in range(len(training_set)):
			sentence = training_set[random.randint(0, len(training_set)-1)]
			gs = preprocess(sentence.x, ws, fs, tag_set)
			y_hat = predict(gs, tag_set, sentence)
			updateWeightsCP(ws, sentence, fs, y_hat)
		print("Epoch done")
		# See how well the model does
		correctnessRate = validate(fs, ws, tag_set, validation_set)
		print("Correctness rate: " + str(correctnessRate))
		print "Time used in epoch: " + str(time.gmtime() - StartTime)
		# Break if it begins to do worse than the previous epoch
		if(correctnessRate <= previousCorrectnessRate): break
		previousCorrectnessrate = correctnessRate
	return previousCorrectnessRate

def validate(fs, ws, tag_set, validation_set):
	numberOfTags = 0
	numberOfCorrectTags = 0
	for sentence in validation_set:
		gs = preprocess(sentence.x, ws, fs, tag_set)
		y_predicted = predict(gs, tag_set, sentence)
		#print("Sentence: " + str(sentence.x))
		#print("y_correct: " + str(sentence.y))
		#print("y_predicted :" + str(y_predicted))
		for (correctTag, predictedTag) in zip(sentence.y, y_predicted):
			numberOfTags += 1
			if predictedTag is correctTag:
				numberOfCorrectTags += 1
	return float(numberOfCorrectTags)/numberOfTags
	
def readFile(filename):
	with open ("./punctuationDataset/" + filename + "Labels.txt") as data:
		labels = data.readlines();
	with open ("./punctuationDataset/" + filename + "Sentences.txt") as data:
		sentences = data.readlines();

	examples = []

	for (sentence, label) in zip(sentences, labels):
		wordsInSentence = sentence.rstrip().split(' ')
		tagsInLabel = label.rstrip().split(' ')
		y = []
		for tag in tagsInLabel:
			if(tag == 'SPACE'): y.append(0)
			if(tag == 'PERIOD'): y.append(1)
			if(tag == 'COMMA'): y.append(2)
			if(tag == 'QUESTION_MARK'): y.append(3)
			if(tag == 'EXCLAMATION_POINT'): y.append(4)
			if(tag == 'COLON'): y.append(5)
		examples.append(Sentence(wordsInSentence, y))
	#random.shuffle(examples)
	return examples

examples = readFile("training")
print("Examples read")
training_set = examples[:1000]
validation_set = examples[1000:1500]
print("Training set and validation set initialized")
fs = initializeFeatureFunctions()
print("Feature functions initialized")
correctnessRate = collinsPerceptron(fs, tag_set, training_set, validation_set)

	
	
	
	
	
	
	