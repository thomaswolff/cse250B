import random
import time
from Templates import *
from FeatureFunctions import *
from math import exp
import copy
	

def initializeFeatureFunctions():
	t = []
	# These are the ones we are using
	t.append(Template1D(-1, True, tag_set, lambda x_index, x, i: i == (len(x) - 1)))
	t.append(Template1D(-1, True, tag_set, lambda x_index, x, i: i < len(x)-1))
	t.append(Template1D(-1, False, tag_set, lambda x_index, x, i: x[i][0].isupper() and len(x[i]) > 1 and i < 2))
	t.append(Template2D(0, True, question_words, tag_set, lambda x_index, x, i: i == (len(x) - 1)))
	t.append(Template2D(-1, False, conjunction_words, tag_set, lambda x_index, x, i: i < (len(x)-1)))

	#t.append(Template2D(-1, True, exclamation_words, tag_set, lambda x_index, x, i: i == (len(x) - 1)))

	# Not using these at the moment
	#t.append(Template2D(0, False, question_words, tag_set, lambda x_index, x, i: i == (len(x) - 1)))
	#t.append(Template1D(-1, True, tag_set, lambda x_index, x, i: x[i][0].isupper() if(x_index is (-1)) else x[0][0].isupper()))
	#t.append(Template1D(-1, False, tag_set, lambda x_index, x, i: x[i][0].isupper() if(x_index is (-1)) else x[0][0].isupper()))
	return t

# Calculate value of a g function given y_prev and y
# y_prev and y are the two tags
# w is the vector of weights
# f is a vector of J feature functions
# x is the given sentence
# i is the position in the sequence and also the number of the g-function
def g(y_prev, y, x, i, ts):
	# sum = 0.0
	# for (weight, function) in zip(w, f):
	# 	if (weight != 0.0):
	# 		sum += weight*function.evaluateLowLevelFeatureFunction(y_prev, y, x, i)
	# return sum
	sum = 0.0
	for t in ts:
		t_value = t(y_prev, y, x, i)
		#print str(t_value)
		if(t_value != minint):
			sum += t_value
	return sum
	
# Calculate all g functions for all y_prev and y given a sentence x
# x is the given sentence, where x is an array of words
# w is the vector of weights
# f is the vector of J feature functions
def preprocess(x, ts, tag_set):
	n = len(x)
	m = len(tag_set)
	gs = [[[0.0 for i in range(m)] for j in range(m)] for k in range(n)]
	i = 0
	while i < n:
		for y_prev in tag_set:
			for y in tag_set:
				gs[i][y_prev][y] = g(y_prev, y, x, i, ts)
		i += 1
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
				dp_table[0][v] = gs[0][START][v]
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
	#print(str(U))
	# Find best prediction for last tag in y
	max_val = minint
	max_u = 1
	for u in range(len(U[n-1])):
		if(U[n-1][u] > max_val):
			max_val = U[n-1][u]
			max_u = u
	y[n-1] = max_u
	# Find best prediction for every other tag in y
	for k in range(n-2, -1, -1):
		max_val = minint
		max_u = 1
		for u in range(len(U[k])):
			u_val = U[k][u] + gs[k+1][u][y[k+1]]
			if(u_val > max_val):
				max_val = u_val
				max_u = u
		y[k] = max_u
	return y

def updateWeightsCP(ts, sentence, y_hat):
	# # For each weight, do the update rule in Collins perceptron
	# for i in range(len(ws)):
	# 	desired = fs[i](sentence.x, sentence.y)
	# 	#print("F of desired: " + str(desired))
	# 	undesired = fs[i](sentence.x, y_hat)
	# 	#print("F of undesired: " + str(undesired))
	# 	ws[i] += (desired - undesired)
	for t in ts:
		#print "Table before update: " + str(t.table)
		t.updateWeights(y_hat, sentence)
		#print "Table after update: " + str(t.table)


def gibbsSample(gs, tag_set, sentence):
	y_star = sentence.y[:]
	while True: 
		for i in range(len(y_star)):
			distribution = getDistributionForTags(gs, tag_set, y_star, i)
			sample = getRandomSampleFromDistribution(distribution, tag_set)
			y_star[i] = sample
		y_star_prob = evaluateProbabilityForY(y_star, gs)
		y_tru_prob = evaluateProbabilityForY(sentence.y, gs)
		#print "Y* prob: " + str(y_star_prob)
		#print "Y true prob: " + str(y_tru_prob)
		if y_star_prob >= evaluateProbabilityForY(sentence.y, gs):
			break
	return y_star

def evaluateProbabilityForY(y, gs):
	probability = 0.0
	for i in range(len(y)):
		if i == 0:
			probability += gs[i][START][y[i]]
		else:
			probability += gs[i][y[i-1]][y[i]]
	return probability

def getRandomSampleFromDistribution(distribution, tag_set):
	x =random.uniform(0, 1)
	cumulative_probability = 0.0
	for (item, item_probability) in zip(tag_set, distribution):
		cumulative_probability += item_probability
		if x < cumulative_probability: break
	return item

def getDistributionForTags(gs, tag_set, y_star, i):
	probabilities = [0.0 for j in range(len(tag_set))]
	denominator = 0.0
	for tag in tag_set:
		if (i == 0):
			denominator += exp(gs[i][START][tag]) * exp(gs[i+1][tag][y_star[i+1]])
		elif (i == len(y_star) - 1):
			denominator += exp(gs[i][y_star[i-1]][tag])
		else:
			denominator += exp(gs[i][y_star[i-1]][tag]) * exp(gs[i+1][tag][y_star[i+1]])
	for j in range(len(tag_set)):
		tag = tag_set[j]
		if (tag == START or tag == STOP):
			probabilities[j] = 0
		else:
			if (i == 0):
				probabilities[j] = exp(gs[i][START][tag]) * exp(gs[i+1][tag][y_star[i+1]])
			elif (i == len(y_star) - 1):
				probabilities[j] = exp(gs[i][y_star[i-1]][tag])
			else:
				probabilities[j] = exp(gs[i][y_star[i-1]][tag]) * exp(gs[i+1][tag][y_star[i+1]])
			probabilities[j] /= denominator
	return probabilities



		
def collinsPerceptron(ts, tag_set, training_set, validation_set, useGibbs):
	# Initialize all weights to be zero
	#ws = [0.0]*len(fs)
	previousCorrectnessRate = 0.0
	iterations = 0
	if (useGibbs):
		print "Using Contrastive divergence to train.."
	else:
		print "Using Collins' perceptron to train.."
	# Epoch loop
	while(True):
		iterations += 1
		epoch_time = 0.0
		epoch_time -= time.time()
		# For each sentence in the training data, update weights
		for i in range(len(training_set)):
			sentence = training_set[random.randint(0, len(training_set)-1)]
			gs = preprocess(sentence.x, ts, tag_set)
			if useGibbs:
				y_hat = gibbsSample(gs, tag_set, sentence)
			else:
				y_hat = predict(gs, tag_set, sentence)
			updateWeightsCP(ts, sentence, y_hat)
		print("Epoch done")
		# See how well the model does
		correctnessRate = validate(ts, tag_set, validation_set)
		epoch_time += time.time()
		print "Time used in epoch: " + str(epoch_time)
		print "Correctness rate: " + str(correctnessRate)
		# Break if it begins to do worse than the previous epoch
		if(correctnessRate <= previousCorrectnessRate): break
		else: 
			previousCorrectnessRate = correctnessRate
			previousModel = copy.deepcopy(ts)
	return previousModel

def validate(ts, tag_set, validation_set):
	numberOfTags = 0
	numberOfCorrectTags = 0
	for sentence in validation_set:
		gs = preprocess(sentence.x, ts, tag_set)
		y_predicted = predict(gs, tag_set, sentence)
		#print("True label: " + str(sentence.y))
		#print("Predicted label: " + str(y_predicted))
		#print("Sentence: " + str(sentence.x))
		#print("y_correct: " + str(sentence.y))
		#print("y_predicted :" + str(y_predicted))
		for (correctTag, predictedTag) in zip(sentence.y, y_predicted):
			numberOfTags += 1
			if predictedTag is correctTag:
				numberOfCorrectTags += 1
	return float(numberOfCorrectTags)/numberOfTags


def validateTestSet(ts, tag_set, validation_set, correctPredictedTags, trueTags):
	numberOfTags = 0
	numberOfCorrectTags = 0
	#trueTags = [0 for i in range(len(tag_set))]
	#correctPredictedTags = [0 for i in range(len(tag_set))]
	totalPredictTime = 0.0
	predict_time = 0.0
	print "True tags: " + str(trueTags)
	for sentence in validation_set:
		predict_time = 0.0
		predict_time -= time.time()
		gs = preprocess(sentence.x, ts, tag_set)
		y_predicted = predict(gs, tag_set, sentence)
		#print("True label: " + str(sentence.y))
		#print("Predicted label: " + str(y_predicted))
		#print("Sentence: " + str(sentence.x))
		#print("y_correct: " + str(sentence.y))
		#print("y_predicted :" + str(y_predicted))
		predict_time += time.time()
		totalPredictTime += predict_time
		for (correctTag, predictedTag) in zip(sentence.y, y_predicted):
			numberOfTags += 1
			trueTags[correctTag] += 1
			if predictedTag is correctTag:
				numberOfCorrectTags += 1
				correctPredictedTags[predictedTag] += 1
	print "True tags after: " + str(trueTags)

	if (trueTags[SPACE] != 0):
		print "Total number of spaces: \t\t" + str(trueTags[SPACE])
		print "Correct predicted spaces: \t\t" + str(correctPredictedTags[SPACE])
		print "Accuracy rate on predicting space: \t" + str(float(correctPredictedTags[SPACE]) / trueTags[SPACE])
		print ""
	if (trueTags[PERIOD] != 0):
		print "Total number of periods: \t\t" + str(trueTags[PERIOD])
		print "Correct predicted periods: \t\t" + str(correctPredictedTags[PERIOD])
		print "Accuracy rate on predicting periods: \t" + str(float(correctPredictedTags[PERIOD]) / trueTags[PERIOD])
		print ""
	if (trueTags[COMMA] != 0):
		print "Total number of commas: \t\t" + str(trueTags[COMMA])
		print "Correct predicted commas: \t\t" + str(correctPredictedTags[COMMA])
		print "Accuracy rate on predicting commas: \t" + str(float(correctPredictedTags[COMMA]) / trueTags[COMMA])
		print ""
	if (trueTags[QUESTION_MARK] != 0):
		print "Total number of question marks: \t\t" + str(trueTags[QUESTION_MARK])
		print "Correct predicted question marks: \t\t" + str(correctPredictedTags[QUESTION_MARK])
		print "Accuracy rate on predicting question marks: \t" + str(float(correctPredictedTags[QUESTION_MARK]) / trueTags[QUESTION_MARK])
		print ""
	if (trueTags[EXCLAMATION_POINT] != 0):
		print "Total number of exclamation points: \t\t" + str(trueTags[EXCLAMATION_POINT])
		print "Correct predicted exclamation points: \t\t" + str(correctPredictedTags[EXCLAMATION_POINT])
		print "Accuracy rate on predicting exclamation points: \t" + str(float(correctPredictedTags[EXCLAMATION_POINT]) / trueTags[EXCLAMATION_POINT])
		print ""
	if (trueTags[COLON] != 0):
		print "Total number of colons: \t\t" + str(trueTags[COLON])
		print "Correct predicted colons: \t\t" + str(correctPredictedTags[COLON])
		print "Accuracy rate on predicting colons: \t" + str(float(correctPredictedTags[COLON]) / trueTags[COLON])
		print ""

	print "Time to predict one sentence: " + str(totalPredictTime / len(validation_set))



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
			if(tag == 'SPACE'): y.append(SPACE)
			if(tag == 'PERIOD'): y.append(PERIOD)
			if(tag == 'COMMA'): y.append(COMMA)
			if(tag == 'QUESTION_MARK'): y.append(QUESTION_MARK)
			if(tag == 'EXCLAMATION_POINT'): y.append(EXCLAMATION_POINT)
			if(tag == 'COLON'): y.append(COLON)
		examples.append(Sentence(wordsInSentence, y))
	random.shuffle(examples)
	return examples


def main():
	numberOfRuns = 10
	run = 0
	correctPredictedTags = [0 for i in range(len(tag_set))]
	trueTags = [0 for i in range(len(tag_set))]
	averageAccuracy = 0
	while run < numberOfRuns:
		run += 1
		examples = readFile("training")
		print("Examples read")
		random.shuffle(examples)
		training_set = examples[:35000]
		validation_set = examples[35000:]
		print("Training set and validation set initialized")
		ts = initializeFeatureFunctions()
		print("Feature functions initialized")
		model = collinsPerceptron(ts, tag_set, training_set, validation_set, True)

		testExamples = readFile("test")
		random.shuffle(testExamples)
		correctnessRate = validateTestSet(model, tag_set, testExamples, correctPredictedTags, trueTags)
		averageAccuracy += correctnessRate
		print ("Correctness rate in test set: " + str (correctnessRate))
	print ""
	print "Finished with " + str(numberOfRuns) + " runs."

	if (trueTags[SPACE] != 0):
		print "Total number of spaces: \t\t" + str(trueTags[SPACE])
		print "Correct predicted spaces: \t\t" + str(correctPredictedTags[SPACE])
		print "Accuracy rate on predicting space: \t" + str(float(correctPredictedTags[SPACE]) / trueTags[SPACE])
		print ""
	if (trueTags[PERIOD] != 0):
		print "Total number of periods: \t\t" + str(trueTags[PERIOD])
		print "Correct predicted periods: \t\t" + str(correctPredictedTags[PERIOD])
		print "Accuracy rate on predicting periods: \t" + str(float(correctPredictedTags[PERIOD]) / trueTags[PERIOD])
		print ""
	if (trueTags[COMMA] != 0):
		print "Total number of commas: \t\t" + str(trueTags[COMMA])
		print "Correct predicted commas: \t\t" + str(correctPredictedTags[COMMA])
		print "Accuracy rate on predicting commas: \t" + str(float(correctPredictedTags[COMMA]) / trueTags[COMMA])
		print ""
	if (trueTags[QUESTION_MARK] != 0):
		print "Total number of question marks: \t\t" + str(trueTags[QUESTION_MARK])
		print "Correct predicted question marks: \t\t" + str(correctPredictedTags[QUESTION_MARK])
		print "Accuracy rate on predicting question marks: \t" + str(float(correctPredictedTags[QUESTION_MARK]) / trueTags[QUESTION_MARK])
		print ""
	if (trueTags[EXCLAMATION_POINT] != 0):
		print "Total number of exclamation points: \t\t" + str(trueTags[EXCLAMATION_POINT])
		print "Correct predicted exclamation points: \t\t" + str(correctPredictedTags[EXCLAMATION_POINT])
		print "Accuracy rate on predicting exclamation points: \t" + str(float(correctPredictedTags[EXCLAMATION_POINT]) / trueTags[EXCLAMATION_POINT])
		print ""
	if (trueTags[COLON] != 0):
		print "Total number of colons: \t\t" + str(trueTags[COLON])
		print "Correct predicted colons: \t\t" + str(correctPredictedTags[COLON])
		print "Accuracy rate on predicting colons: \t" + str(float(correctPredictedTags[COLON]) / trueTags[COLON])
		print ""

	print "Average accuracy \t\t\t" + str(averageAccuracy / float(numberOfRuns))

	
main()
	
	
	
	
	