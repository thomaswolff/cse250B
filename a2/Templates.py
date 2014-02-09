from Constants import *

class Template1D(object):
	x_index = -1
	use_y_i = True
	premise = lambda x: True
	table = []
	dim = 1
	set_of_b = []

	def __init__(self, x_index, use_y_i, set_of_b, premise):
		self.x_index = x_index
		self.use_y_i = use_y_i
		self.premise = premise
		self.set_of_b = set_of_b
		self.table = [0.0 for b in set_of_b]

	def __call__(self, y_prev, y, x, i):
		if self.premise(self.x_index, x, i):
			if self.use_y_i:
				return self.table[y]
			else: 
				return self.table[y_prev]
		else:
			return minint

	def updateWeights(self, y_hat, sentence):
		x = sentence.x
		y_true = sentence.y
		w_true_Table = [[0.0 for i in range(len(x))] for b in self.set_of_b]
		f_true_Table = [False for b in self.set_of_b]
		w_hat_Table = [[0.0 for i in range(len(x))] for b in self.set_of_b] 
		f_hat_Table = [False for b in self.set_of_b]
		for i in range(len(x)):
			w_true = 0.0
			w_predicted = 0.0
			if i == 0:
				w_true = self(START, y_true[i], x, i)
				if(w_true != minint):
					if self.use_y_i:
						w_true_Table[y_true[i]][i] = 1
						f_true_Table[y_true[i]] = True
					else:
						w_true_Table[START][i] = 1
						f_true_Table[START] = True
				w_predicted = self(START, y_hat[i], x, i)
				if(w_predicted != minint):
					if self.use_y_i:
						w_hat_Table[y_hat[i]][i] = 1
						f_hat_Table[y_hat[i]] = True
					else:
						w_hat_Table[START][i] = 1
						f_hat_Table[START] = True
			else:
				w_true = self(y_true[i-1], y_true[i], x, i)
				if(w_true != minint):
					if self.use_y_i:
						w_true_Table[y_true[i]][i]  = 1
						f_true_Table[y_true[i]] = True
					else:
						w_true_Table[y_true[i-1]][i]  = 1
						f_true_Table[y_true[i-1]] = True
				w_predicted = self(y_hat[i-1], y_hat[i], x, i)
				if(w_predicted != minint):
					if self.use_y_i:
						w_hat_Table[y_hat[i]][i] = 1
						f_hat_Table[y_hat[i]] = True
					else:
						w_hat_Table[y_hat[i-1]][i] = 1
						f_hat_Table[y_hat[i-1]] = True
		for i in range(len(f_true_Table)):
			true_sum = 0.0
			predicted_sum = 0.0
			if(f_true_Table[i]):
				# sum corresponding row in w_true_table
				for a in w_true_Table[i]:
					true_sum += a
			if(f_hat_Table[i]):
				# sum corresponding row in w_hat_table
				for a in w_hat_Table[i]:
					predicted_sum += a
			if(true_sum != predicted_sum):
				# update weight
				self.table[i] += true_sum - predicted_sum

class Template2D(object):
		x_index = -1
		use_y_i = True
		premise = lambda x: True
		table = []
		set_of_a = {}
		set_of_b = []
		dim = 2

		def __init__(self, x_index, use_y_i, set_of_a, set_of_b, premise):
			self.x_index = x_index
			self.use_y_i = use_y_i
			self.premise = premise
			self.set_of_a = set_of_a
			self.set_of_b = set_of_b
			self.table = [[0.0 for a in range(len(set_of_a))] for b in range(len(set_of_b))]

		def __call__(self, y_prev, y, x, i):
			word = ""
			if self.x_index == (-1):
				word = x[i] 
			else: 
				word = x[self.x_index]
			word = word.lower()
			#print "Length of word: " + str(len(x))
			#print "i: " + str(i)
			#print "word: " + word
			if self.premise(self.x_index, x, i) and word in self.set_of_a:
				a_index = self.set_of_a[word]
				if self.use_y_i:
					return self.table[y][a_index]
				else:
					return self.table[y_prev][a_index]
			else:
				return minint

		def updateWeights(self, y_hat, sentence):
			x = sentence.x
			y_true = sentence.y
			w_true_Table = [[[0.0 for i in range(len(x))] for a in range(len(self.set_of_a))] for b in range(len(self.set_of_b))]
			f_true_Table = [[False for a in range(len(self.set_of_a))] for b in range(len(self.set_of_b))]
			w_hat_Table = [[[0.0 for i in range(len(x))] for a in range(len(self.set_of_a))] for b in range(len(self.set_of_b))]
			f_hat_Table = [[False for a in range(len(self.set_of_a))] for b in range(len(self.set_of_b))]
			for i in range(len(x)):
				w_true = 0.0
				w_predicted = 0.0
				word = ""
				if self.x_index == (-1):
					word = x[i] 
				else: 
					word = x[self.x_index]
				word = word.lower()
				if word in self.set_of_a:
					a_index = self.set_of_a[word]
					if i == 0:
						w_true = self(START, y_true[i], x, i)
						#print "w_true: " + str(w_true)
						if(w_true != minint):
							#print "Hit true label"
							#print "length of word: " + str(len(x))
							#print "i: " + str(i)
							if self.use_y_i:
								w_true_Table[y_true[i]][a_index][i] = 1
								f_true_Table[y_true[i]][a_index] = True
							else:
								w_true_Table[START][a_index][i] = 1
								f_true_Table[START][a_index] = True
						w_predicted = self(START, y_hat[i], x, i)
						#print "w_predicted: " + str(w_predicted)
						if(w_predicted != minint):
							#print "length of word: " + str(len(x))
							#print "i: " + str(i)
							if self.use_y_i:
								w_hat_Table[y_hat[i]][a_index][i] = 1
								f_hat_Table[y_hat[i]][a_index] = True
							else:
								w_hat_Table[START][a_index][i] = 1
								f_hat_Table[START][a_index] = True
					else:
						w_true = self(y_true[i-1], y_true[i], x, i)
						#print "w_true: " + str(w_true)
						if(w_true != minint):
							#print "length of word: " + str(len(x))
							#print "i: " + str(i)
							#print "Hit true label"
							if self.use_y_i:
								w_true_Table[y_true[i]][a_index][i]  = 1
								f_true_Table[y_true[i]][a_index] = True
							else:
								w_true_Table[y_true[i-1]][a_index][i]  = 1
								f_true_Table[y_true[i-1]][a_index] = True
						w_predicted = self(y_hat[i-1], y_hat[i], x, i)
						#print "w_predicted: " + str(w_predicted)
						if(w_predicted != minint):
							#print "length of word: " + str(len(x))
							#print "i: " + str(i)
							if self.use_y_i:
								w_hat_Table[y_hat[i]][a_index][i] = 1
								f_hat_Table[y_hat[i]][a_index] = True
							else:
								w_hat_Table[y_hat[i-1]][a_index][i] = 1
								f_hat_Table[y_hat[i-1]][a_index] = True
			#print "Sentence: " + str(x)
			#print "True label: " + str(y_true)
			#print "Predicted label: " + str(y_hat)
			#print "True f table: " + str(f_true_Table)
			#print "Predicted f table: " + str(f_hat_Table)
			for i in range(len(self.set_of_b)):
				for j in self.set_of_a:
					a_index = self.set_of_a[j]
					true_sum = 0.0
					predicted_sum = 0.0
					if(f_true_Table[i][a_index]):
						# sum corresponding row in w_true_table
						#print "Hit true"
						#print "True: b: " +str(i) + ", j: " + str(j) + ", i-row: "+  str(w_true_Table[i][a_index])
						for a in w_true_Table[i][a_index]:
							true_sum += a
					if(f_hat_Table[i][a_index]):
						# sum corresponding row in w_hat_table
						#print "Hit predicted"
						#print "Predicted: b: " +str(i) + ", j: " + str(j) + ", i-row: " + str(w_hat_Table[i][a_index])
						for a in w_hat_Table[i][a_index]:
							predicted_sum += a
					#print "True sum: " + str(true_sum)
					#print "Predicted sum: " + str(predicted_sum)
					if(true_sum != predicted_sum):
						#print "update weight"
						self.table[i][a_index] += true_sum - predicted_sum
			#print "Table in update" + str(self.table)