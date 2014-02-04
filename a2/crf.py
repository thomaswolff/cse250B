import random

START = 6
SPACE = 0
PERIOD = 1
COMMA = 2
QUESTION_MARK = 3
EXCAMLATION_POINT = 4
COLON = 5
STOP = 7

minint = -100000000000

question_words = {"how", "where", "who", "why", "when"}
conjunctions_words = {"and", "or", "but", "for", "as", "because", "nor", "yet", "so"}

class Sentence(object):
	x = []
	y = []
	predicted_y = []

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.predicted_y = []
		
def featureFunction1(y_prev, y, x, i):
	return y is QUESTION_MARK and x[0].lower() in question_words and i == len(x) - 1


def fFsQuestionTemplate(k):
	def f(y_prev, y, x, i):
		return y is QUESTION_MARK and x[k].lower() in question_words and i == len(x) - 1
	return f

def fFsEndsWithTemplate(k, suffix):
	def f(y_prev, y, x, i):
		return x[k].endswith(suffix)
	return f

def fFsStartWithCapitalLetter(k):
	def f(y_prev, y, x, i):
		return x[k][0].isupper()
	return f

def fFsStartssWithTemplate(k, suffix):
	def f(y_prev, y, x, i):
		return x[k].startswith(suffix)
	return f

def fFsConjunction(k):
	def f(y_prev, y, x, i):
		return y is COMMA and x[k].lower() in conjunctions_words
	return f

def fFsPeriod(k):
	def f(y_prev, y, x, i):
		return k is i and y is PERIOD
	return f





	
def featureFunction2(y_prev, y, x, i):
	return y is STOP

# Calculate value of a g function given y_prev and y
# y_prev and y are the two tags
# w is the vector of weights
# f is a vector of J feature functions
# x is the given sentence
# i is the position in the sequence and also the number of the g-function
def g(y_prev, y, w, f, x, i):
	sum = 0.0
	for (weight, function) in zip(w, f):
		sum += weight*function(y_prev, y, x, i)
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
def U(k, v, gs, tag_set, dp_table):
	max_val = minint
	print "k: " + str(k)
	if(k is 1):
		print "er i if"

		for u in tag_set:
			print "u: " + str(u)
			if(dp_table[k-1][u] is minint):
				u_val = gs[k-1][u][v]
			else:
				u_val = dp_table[k-1][u]
			if (u_val > max_val):
				max_val = u_val
		dp_table[0][v] = max_val

	else:
		print "else"
		print "u: "
		for u in tag_set:
			if(dp_table[k-1][u] is minint):
				u_val = U(k-1, u, gs, tag_set, dp_table) + gs[k-1][u][v]
			else:
				u_val = dp_table[k-1][u]
			if (u_val > max_val):
				max_val = u_val
		dp_table[k-1][v] = max_val
	return max_val

# Fill the U matrix by calling U on every tag in the tag set
def fill_U_matrix(gs, tag_set, n):
	dp_table = [[minint for j in range(len(tag_set))] for i in range(n)]
	for tag in tag_set:
		U(n, tag, gs, tag_set, dp_table)
	return dp_table

# Predict the best tag sequence y for sentence x
def predict(gs, tag_set, x):
	# Initialize n and y
	n = len(x)
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
	
def perceptron():

	train()
	# Train
	
	
#def validate():
	# Preprocess
	# Predict
	
def readFile(filename):
	with open ("./punctuationDataset/" + filename + "Labels.txt") as data:
		labels = data.readlines();
	with open ("./punctuationDataset/" + filename + "Sentences.txt") as data:
		sentences = data.readlines();

	examples = []

	for (sentence, label) in zip(sentences, labels):
		wordsInSentence = sentence.rstrip().split(' ')
		tagsInLabel = label.rstrip().split(' ')
		examples.append(Sentence(wordsInSentence, tagsInLabel))
	#random.shuffle(examples)
	return examples

examples = readFile("training")
e = examples[19]
x = e.x
y = e.y
#print("Sentence: " + str(x))
#print("Label: " + str(y))
w = [10.0]
f = [featureFunction1]
tag_set = [SPACE, PERIOD, COMMA, QUESTION_MARK, EXCAMLATION_POINT, COLON]
gs = preprocess(x, w, f, tag_set)
#for g in gs:
#	print("-------G-------")
#	for y_prev in tag_set:
#		row = ""
#		for y in tag_set:
#			row += " " + str(g[y_prev][y])
#		print(row)
#	print("---------------")
predictLabel = predict(gs, tag_set, x)
print predictLabel

A = {}
B = {}
fj = []
for a in A:
	for b in B:
		fj = createFeatureFunc(a, b)
	
	
	
	
	
	
	