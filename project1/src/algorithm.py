import math
import random

class Example(object):
	classification = 0
	features = []
	predicted = 0

	def __init__(self, classification, features):
		self.classification = classification
		self.features = features
		self.predicted = 0


def train(examples, beta, lamd, mu):
	example = examples[random.randint(0, len(examples) - 1)]
	index = 0
	example.predicted = classify(example)
	for feature in example.features:
		beta[index] = beta[index] + lamd *((example.classification - example.predicted) * feature - 2 * mu * beta[index])
		index += 1

def classify(example):
	l = 0
	index = 0
	for f in example.features:
		c = beta[index]
		l += c * f
		index += 1
	temp = 0
	try:
		 temp = 1.0 / (1.0 + math.exp(-l))
	except:
		temp = 1
	return temp

def validate(examples):
	random.shuffle(examples)
	wrong = 0
	sum = 0.0
	for example in examples:
		example.predicted = classify(example)
		if (example.classification == 1):
			sum += example.predicted
		else:
			sum += (1 - example.predicted)
		if (example.predicted > 0.5):
			example.predicted = 1
		else:
			example.predicted = 0
		
		if example.classification != example.predicted:
			wrong += 1
	return sum / len(examples)

def test():
	examples = readData("test")
	wrong = 0
	for example in examples:
		p = classify(example)
		if p > 0.5:
			example.predicted = 1
		else:
			example.predicted = 0
		if example.predicted != example.classification:
			wrong += 1
	return wrong / float(len(examples))

def grid(examples):
	lamdas = [0.00002]
	mus = [0.002]

	lbest = 0
	mubest = 0
	bestLGL = 0

	for l in lamdas:
		for m in mus:
			beta = [0] * 800
			examples = readData("train")
			temp = 0
			temp = sdg(examples, beta, l, m)
			beta = [0] * 800
			examples = readData("train")
			temp += sdg(examples, beta, l, m)
			temp = temp / 2.0
			if temp > bestLGL:
				lbest = l
				mubest = m
	print "Best lamda: " + str(lbest)
	print "Best mu: " + str(mubest) 

def setPredictedToZero(examples):
	for example in examples:
		example.predicted = 0


def readData(filename):
	with open ("../data/" + filename) as data:
		content = data.readlines();

	examples = []

	for line in content:
		rawLine = line.split(' ')
		classification = int(rawLine[0])
		if classification < 0:
			classification = 0
		rawLine = rawLine[1::]
		features = []
		index = 0
		for feature in rawLine:
			temp = feature.split(':')
			if len(temp) == 2:
				features.append(int(temp[1]))
				index += 1
		examples.append(Example(classification, features))
	random.shuffle(examples)
	return examples

def sdg(examples, beta, lamd, mu):
	prev = 0
	next = validate(examples[300:])
	print next
	converge = 0
	epochs = 0
	convergeRate = 0.000000001
	iterations = 0

	while epochs < 10:
		if iterations % 300 == 0:
			next = validate(examples[500:])
			print next
			epochs += 1
		lamd = lamd * 0.999
		if (math.fabs(next - prev)) >  convergeRate: 
			converge += 1
		else:
			prev = next
			converge = 0

		train(examples[:500], beta, lamd, mu)
		iterations += 1
		
	print next
	print "Done."
	return next


examples = readData("train")

beta = [0] * 800
lamd = 0.00003
mu = 0.1

train(examples[:300], beta, lamd, mu)
sdg(examples, beta, lamd, mu)
error_rate = test()
print "Error rate: " + str(error_rate)
























