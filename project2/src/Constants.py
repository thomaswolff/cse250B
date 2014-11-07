START = 0
SPACE = 1
PERIOD = 2
COMMA = 3
QUESTION_MARK = 4
EXCLAMATION_POINT = 5
COLON = 6
STOP = 7
ounter = 0
minint = -100000

question_words = {"how" : 0, "where" : 1, "who" : 2, "why" : 3, "when" : 4, "what" : 5, "can" : 6}
conjunction_words = {"and" : 0, "or" : 1, "but" : 2, "for" : 3, "as" : 4, "because" : 5, "nor" : 6, "yet" : 7, "so" : 8}
exclamation_words = {"now" : 0, "get" : 1}

indictment_words = {"hi" : 0, "hello" : 1, "hey" : 3}
tag_set = [START, SPACE, PERIOD, COMMA, QUESTION_MARK, EXCLAMATION_POINT, COLON, STOP]

class Sentence(object):
	x = []
	y = []
	predicted_y = []

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.predicted_y = []