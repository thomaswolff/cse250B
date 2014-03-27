import numpy
import math
import moviereviews
import random


RIGHT = 1
LEFT = 2
ROOT = 0

Z = 3
P = 4

def softmax(z):
	denominator = reduce(lambda x, y: x + y, map(lambda x: math.exp(x), z))
	return map(lambda x: (math.exp(x)) / denominator, z)

def softmaxDerivative(z):
	soft = softmax(z)
	return [soft[i]*(1-soft[i]) for i in range(len(z))]


class RAE:
	def __init__(self, mapIdToWord, lexicon, W, U, V, trainingReviews, validationReviews, testReviews, D, R, alpha, lambd, mu):
		self.lexicon = lexicon
		self.mapIdToWord = mapIdToWord
		self.W = W
		self.dW = None
		self.U = U
		self.dU = None
		self.V = V
		self.dV = None
		self.D = D
		self.R = R
		self.example = None
		self.alpha = alpha
		self.lambd = lambd
		self.mu = mu
		self.trainingReviews = trainingReviews
		self.validationReviews = validationReviews
		self.testReviews = testReviews

		self.randomPhrase = None

	def toLatexTree(self, node):
		temp = ""
		if (node.leftChild != None and node.rightChild != None):
			temp += "[ "
			temp += self.toLatexTree(node.leftChild)
			temp += self.toLatexTree(node.rightChild)
			temp += "]"
		else:
		 	temp += "[ \\textit{" + str(self.mapIdToWord[node.phrase[0]]) + "} ] "
		return temp



	def train(self):
		while(True):
			for review in self.trainingReviews:
				for sentence in review.sentences:
					self.example = sentence
					#bygge trestruktur
					T = self.findTreeStructure(sentence.words)
					# feedForward(T)
					# backprop
					self.backpropagation(T)
					self.randomPhrase = T
					print "\Tree" + self.toLatexTree(T)
					#print " ----------------------------\n"

					#oppdater parametre
					self.W += self.lambd * (self.dW - (0.5*self.mu*self.W))
					self.U += self.lambd * (self.dU - (0.5*self.mu*self.U)) 
					self.V += self.lambd * (self.dV - (0.5*self.mu*self.V))
					#(dW, dU, dV) = self.calculateNumericalDerivatives(T, 0.0000000001)
					# print "----------------------- dW -----------------------"
					# print "Backprop:"
					# print self.dW
					# print "Numerical"
					# print dW

					# print "----------------------- dU -----------------------"
					# print "Backprop:"
					# print self.dU
					# print "Numerical"
					# print dU

					# print "----------------------- dV -----------------------"
					# print "Backprop:"
					# print self.dV
					# print "Numerical"
					# print dV
					break
				break
			break
			print self.validate(self.validationReviews)

	def validate(self, reviews):
		numberOfCorrectReviews = 0
		mostPositivePhrases = [None for i in range(10)]
		mostNegativePhrases = [None for i in range(10)]

		mostSimilarPhrase = None


		for review in reviews:
			count = 0
			for sentence in review.sentences:
				root = self.findTreeStructure(sentence.words)
				count += root.p.output[0]
				self.addPhrase(mostPositivePhrases, root, lambda x, y: x > y)
				self.addPhrase(mostNegativePhrases, root, lambda x, y: x < y)

				similarity = self.euclid(root, mostSimilarPhrase)
				if mostSimilarPhrase is None or similarity < mostSimilarPhrase[1]:
					mostSimilarPhrase = (root, similarity)

				#print "p: ", root.p.output[0]
			if count > 0 and review.label[0] is 1:
					numberOfCorrectReviews += 1
			elif count < 0 and review.label[0] is 0:
					numberOfCorrectReviews += 1
			# elif count is len(review.sentences)/2:
			# 	randomLabel = random.randint(0, 1)
			# 	numberOfCorrectReviews += randomLabel
		print "_____________POSITIVE_________________"
		for positive, output in mostPositivePhrases:
			print self.phraseToString(positive), output
		print "--------------------------------------"
		print ""
		print "_____________NEGATIVE_________________"
		for negative, output in mostNegativePhrases:
			print self.phraseToString(negative), output
		print ""
		print "______A RANDOM PHRASE_________________"
		print self.phraseToString(self.randomPhrase.phrase)
		print "______A SIMILAR PHRASE________________"
		print self.phraseToString(mostSimilarPhrase[0].phrase)

		return numberOfCorrectReviews/float(len(reviews))

	def euclid(self, this, best):
		if best is None:
			return 1000
		thisEuclid = 0
		sum = 0
		for i in range(len(this.output)):
			sum += (this.output[i] - self.randomPhrase.output[i]) * (this.output[i] - self.randomPhrase.output[i])
		return math.sqrt(sum) < best




	def phraseToString(self, phrase):
		out = ""
		for word in phrase:
			out += self.mapIdToWord[word] + " "
		return out

	def addPhrase(self, phrases, node, f):
		most = self.findPhrase(node, (node.phrase, node.p.output[0]), f)
		for i in range(len(phrases)):
			if phrases[i] == None:
				phrases[i] = most
				break
		for i in range(len(phrases)):
			if phrases[i] != None and f(most[1], phrases[i][1]):
				phrases[i] = most
				break

	def findPhrase(self, node, most, f):
		if node.leftChild != None and node.rightChild != None:
			most = self.findPhrase(node.leftChild, most, f)
			most = self.findPhrase(node.rightChild, most, f)
			if f(node.p.output[0], most[1]):
				most = (node.phrase, node.p.output[0])
		return most

	#data = List of words in a sentence
	def findTreeStructure(self, data):
		fringe = []
		for word in data:
			node = NoneOutputNode(self.lexicon[word], phrase=[word])
			node.a=node.output
			fringe.append(node)

		while(len(fringe) > 1):
			reconstructionLosses = []
			for i in range(len(fringe)-1):
				p = OutputNode(numpy.zeros(self.D), Node(self.example.label), nodeType = P, loss=lambda x, y: (1-self.alpha)*self.l2Norm(x, y), 
				lossDerivative=lambda x, y: (1-self.alpha)*2*(x-y))
				parent = NoneOutputNode(numpy.zeros(self.D), p = p, leftChild=fringe[i], 
				rightChild=fringe[i+1])
				parent.output = self.calculateMeaningVector(parent)
				p.child = parent
				reconstructionLosses.append((self.e1(parent), parent, i))

			minimum = reduce(lambda x, y: x if (x[0][2] < y[0][2]) else y, reconstructionLosses)
			zi = minimum[0][0]
			zj = minimum[0][1]
			newNode = minimum[1]
			#newNode.h = tanh
			#newNode.hDerivate = tanhD
			newNode.p.output = self.calculateLabelVector(newNode.p)
			z1 = OutputNode(zi, newNode.leftChild, nodeType = Z, position=LEFT, child = newNode)
			z2 = OutputNode(zj, newNode.rightChild, nodeType = Z, position=RIGHT, child = newNode)
			activation = numpy.dot(self.U, numpy.concatenate((newNode.output, [1])))
			z1.a = activation[0:self.D]
			z2.a = activation[self.D:]
			newNode.z1 = z1
			newNode.z2 = z2
			newNode.parents.append(z1)
			newNode.parents.append(z2)
			index = minimum[2]


			#Need to set up the loss and h functions correctly

			newNode.phrase = newNode.leftChild.phrase + newNode.rightChild.phrase
			newNode.numberOfWords = len(newNode.phrase)
			newNode.leftChild.parents.append(newNode)
			newNode.rightChild.parents.append(newNode)
			newNode.leftChild.position = LEFT
			newNode.rightChild.position = RIGHT
			fringe.pop(index)
			fringe.pop(index)
			fringe.insert(index, newNode)
		return fringe[0]

	def e1(self, xk):
		ni = xk.leftChild.numberOfWords
		nj = xk.rightChild.numberOfWords
		n = ni + nj
		(zi, zj) = self.calculateReconstruction(xk)
		xi = xk.leftChild.output
		xj = xk.rightChild.output
		return (zi, zj, self.alpha*((ni / float(n)) * self.l2Norm(xi, zi) + (nj / float(n)) * self.l2Norm(xj, zj)))


	def l2Norm(self, xi, zi):
		return numpy.sum(map(lambda x: x * x, xi - zi))

	def calculateReconstruction(self, xk):
		z = numpy.dot(self.U, numpy.concatenate((xk.output, [1])))
		return (z[:(len(z) / 2)], z[(len(z) / 2) :])

	def calculateMeaningVector(self, node):
		xi = node.leftChild.output
		xj = node.rightChild.output
		a = numpy.dot(self.W, numpy.concatenate((xi, xj, [1])))
		node.a = a
		return node.h(a)

	def calculateLabelVector(self, node):
		a = numpy.dot(self.V, node.child.output)
		node.a = a
		return node.h(a)

	def feedForward(self, node, J):
		if node.leftChild != None and node.rightChild != None:
			J += self.feedForward(node.leftChild, J)
			J += self.feedForward(node.rightChild, J)

			# Calculate own meaning vector
			node.output = self.calculateMeaningVector(node)
			# Calculate reconstruction loss
			J += self.e1(node)[2]
			# Calculate prediction loss
			node.p.output = self.calculateLabelVector(node.p)
			J += node.p.loss(node.p.output, node.p.comparisonNode.output)
		return J
		#Add else here if wanting to change meaning vector of leaf nodes

	def calculateDeltaOutput(self, node):
		hDerivate = node.hDerivate
		a = node.a
		o = node.output
		c = node.comparisonNode.output
		h_of_a = hDerivate(a)
		if node.nodeType is P:
			node.delta = [node.lossDerivative(o[i], c[i])*h_of_a[i] for i in range(len(o))]
		elif node.nodeType is Z:
			parent = node.child
			ni = parent.leftChild.numberOfWords
			nj = parent.rightChild.numberOfWords
			n = ni + nj
			if node.position is LEFT:
				n = ni / float(n)
			elif node.position is RIGHT:
				n = nj / float(n)
			node.delta = [2*n*self.alpha*(o[i]-c[i])*h_of_a[i] for i in range(len(o))]


	def calculateDelta(self, node):
		hDerivative = map(node.hDerivate, node.a)
		delta = numpy.zeros(self.D)
		W = None
		for parent in node.parents:
			if (isinstance(parent, OutputNode)):
				if parent.nodeType is Z:
					if parent.position is LEFT:
						W = self.U[0:self.D, 0:self.D]
					elif parent.position is RIGHT:
						W = self.U[self.D:, 0:self.D]
				elif parent.nodeType is P:
					W = self.V
			else:
				W = self.W
				if node.position is LEFT:
					W = W[:, 0:(self.D)]
				elif node.position is RIGHT:
					W = W[:, (self.D):2*self.D]
			delta = delta + (numpy.dot(parent.delta, W))
		delta *= hDerivative
		node.delta = delta


	def calculateOutputDeltaVectors(self, node, queue = []):
		if node.leftChild != None and node.rightChild != None:
			self.calculateDeltaOutput(node.z1)
			self.calculateDeltaOutput(node.z2)
			self.calculateDeltaOutput(node.p)
			queue.append(node.leftChild)
			queue.append(node.rightChild)
		if len(queue) > 0:
			self.calculateOutputDeltaVectors(queue.pop(0))

	def calculateDeltaVectors(self, node, queue = []):
		if node.leftChild != None and node.rightChild != None:
			queue.append(node.leftChild)
			queue.append(node.rightChild)
		# Calculate delta vector according to equation 4
		self.calculateDelta(node)
		if len(queue) > 0:
			self.calculateDeltaVectors(queue.pop(0))


	def backpropagation(self, root):
		# Compute the delta vector for each output node
		# Working backwards, compute the delta vector of each non-output node using Equation 4
		# 
		# Traverse tree in breadth first order
		# For each internal node
		# 	Compute delta vector for z1, z2 and p
		# For each leaf node
		# 	Compute delta vector for p
		self.calculateOutputDeltaVectors(root)

		# Traverse tree in breadth first order
		# For each non-output node calculate its delta vector using equation 4
		self.calculateDeltaVectors(root)

		# For every node accumulate its weights contribution to the total loss function
		# Zero out derivative matrices
		self.dW = numpy.zeros((self.D, 2*self.D + 1))
		self.dU = numpy.zeros((2*self.D, self.D + 1))
		self.dV = numpy.zeros((self.R, self.D))
		self.calculateDerivatives(root)

	def calculateDerivatives(self, node):
		if node.leftChild != None and node.rightChild != None:
			self.calculateDerivatives(node.leftChild)
			self.calculateDerivatives(node.rightChild)
			self.calculateDerivative(node)

	def calculateDerivative(self, node):
		# Calculate z1, z2, p and self
		self.dU += numpy.outer(numpy.concatenate((node.z1.delta, node.z2.delta)), numpy.concatenate((node.output, [1])))
		self.dV += numpy.outer(node.p.delta, node.output)
		self.dW += numpy.outer(node.delta, numpy.concatenate((node.leftChild.output, node.rightChild.output, [1])))

	def calculateNumericalDerivatives(self, root, epsilon):
		dW = numpy.zeros((self.D, 2*self.D + 1))
		dU = numpy.zeros((2*self.D, self.D + 1))
		dV = numpy.zeros((self.R, self.D))
		for i in range(self.D):
			for j in range(2*self.D + 1):
				self.W[i][j] += epsilon
				J1 = self.feedForward(root, 0)
				self.W[i][j] -= 2*epsilon
				J2 = self.feedForward(root, 0)
				dW[i][j] = (J1-J2)/(2.0*epsilon)
				self.W[i][j] += epsilon

		for i in range(2*self.D):
			for j in range(self.D + 1):
				self.U[i][j] += epsilon
				J1 = self.feedForward(root, 0)
				self.U[i][j] -= 2*epsilon
				J2 = self.feedForward(root, 0)
				dU[i][j] = (J1-J2)/(2.0*epsilon)
				self.U[i][j] += epsilon

		for i in range(self.R):
			for j in range(self.D):
				self.V[i][j] += epsilon
				J1 = self.feedForward(root, 0)
				self.V[i][j] -= 2*epsilon
				J2 = self.feedForward(root, 0)
				dV[i][j] = (J1-J2)/(2.0*epsilon)
				self.V[i][j] += epsilon
		return (dW, dU, dV)

def tanh(a):
	return [(math.exp(a[i])-math.exp(a[i]))/(math.exp(a[i])+math.exp(a[i])) for i in range(len(a))]

def tanhD(a):
	return 1 - tanh([a])[0]*tanh([a])[0]

class  Node(object):
	def __init__(self, output,h = lambda x: x, hDerivate = lambda x: 1):
		self.output = output
		self.h = h
		self.hDerivate = hDerivate
		self.a = None
		self.delta = None

class NoneOutputNode(Node):
	def __init__(self, output, p = None, z1 = None, z2 = None, 
		leftChild = None, rightChild = None, position = ROOT,
		h = lambda x: x, hDerivate = lambda x: 1, numberOfWords = 1, phrase=[]):
		super(NoneOutputNode, self).__init__(output, h, hDerivate)
		self.leftChild = leftChild
		self.rightChild = rightChild
		self.numberOfWords = numberOfWords
		self.z1 = z1
		self.z2 = z2
		self.p = p
		self.parents = []
		self.position = position
		self.phrase = phrase
		if z1 != None:
			self.parents.append(z1)
		if z2 != None:
			self.parents.append(z2)
		if p != None:
			self.parents.append(p)

	def __str__(self):
		return str(self.toString())

	def toString(self, output=[], queue=[]):
		output.append(self.phrase)
		if self.rightChild != None and self.leftChild != None:
			queue.append(self.leftChild)
			queue.append(self.rightChild)
		if len(queue) > 0:
			queue.pop(0).toString(output, queue)
		return output

class OutputNode(Node):
	def __init__(self, output, comparisonNode, h = lambda x: x, hDerivate = lambda x: numpy.ones(len(x)),
		child = None, loss = lambda x, y: self.l2Norm(x, y), 
		lossDerivative = lambda x, y: 2*(x-y), nodeType = None, position = ROOT):
		super(OutputNode, self).__init__(output, h, hDerivate)
		self.loss = loss
		self.lossDerivative = lossDerivative
		self.child = child
		self.comparisonNode = comparisonNode
		self.nodeType = nodeType
		self.position = position

def createLexicon(reviews, sigma, d):
	lexicon = {}
	for review in reviews:
		for sentence in review.sentences:
			for word in sentence.words:
				if word not in lexicon:
					lexicon[word] = numpy.random.normal(0.0, sigma, d)
	return lexicon


def main():
	DEBUG = False
	alpha = 0.5
	lambd = 0.0001
	mu = 0.001
	sigma = 2.0
	d = 2
	r = 1
	W = numpy.zeros((d, 2*d+1)) + 0.1
	U = numpy.zeros((2*d, d+1)) + 0.1
	V = numpy.zeros((r, d)) + 0.1

	if DEBUG:
		mapIdToWord = {"hei" : "hei", "how": "how"}
		testSentence = moviereviews.Sentence(["hei", "how"], [0], mapIdToWord)
		
		debugReviews = [moviereviews.Review([testSentence], [0])]
		testLexicon = {}
		for word in testSentence.words:
			testLexicon[word] = numpy.random.normal(0.0, sigma, d)
		rae = RAE(mapIdToWord, testLexicon, W, U, V, debugReviews, debugReviews, debugReviews, d, r, alpha, lambd, mu)

	else:
		dataset = moviereviews.Dataset(sigma, d)
		reviews = dataset.reviews
		random.shuffle(reviews)
		trainingReviews = reviews[:5]
		validationReviews = reviews[500:505]
		testReviews = reviews[1500:2000]
		lexicon = dataset.lexicon	
		rae = RAE(dataset.mapIdToWord, lexicon, W, U, V, trainingReviews, validationReviews, testReviews, d, r, alpha, lambd, mu)


	rae.train()

if __name__ == "__main__":
    main()










