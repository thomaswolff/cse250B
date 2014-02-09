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