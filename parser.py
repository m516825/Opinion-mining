import jieba
from hanziconv import HanziConv

new_words = ['服務', '環境', '價格', '交通', '餐廳', '態度', '人員', '客房', '設備', '空調', '房價', '地理', '早餐', '性價比']

class Parser(object):
	def __init__(self):
		self.add_words()
	def add_words(self):
		for w in new_words:
			jieba.add_word(w, freq=100000000)
	def parse(self, sentence):
		# s_sentence = HanziConv.toSimplified(sentence)
		# return [HanziConv.toTraditional(w) for w in jieba.cut(s_sentence)]
		return [w for w in jieba.cut(sentence)]

if __name__ == '__main__':
	parser = Parser()
	s = "地理位置服務環境"
	print(parser.parse(s))