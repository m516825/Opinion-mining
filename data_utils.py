import sys
import numpy as np
import tensorflow as tf
from parser import Parser
from tensorflow.python.platform import gfile
import collections
import _pickle as cPickle
import math
import re
try:
	import cPickle as pickle
except ImportError:
	import pickle

min_count = 3
max_seqlen = 50

class Data(object):
	def __init__(self, train_x, labels, aspect_idx, dev_size, g_train, g_label):
		train_x, labels = self.suffle_data(train_x, labels)
		(use_Pdist, use_Ndist, neg_label, pos_label, sent_Pdist, sent_Ndist, polarity) = labels
		self.dev_x = train_x[:dev_size]
		self.train_x = train_x[dev_size:]

		self.dev_use_Pdist = use_Pdist[:dev_size]
		self.use_Pdist = use_Pdist[dev_size:]
		self.dev_use_Ndist = use_Ndist[:dev_size]
		self.use_Ndist = use_Ndist[dev_size:]
		self.dev_neg_label = neg_label[:dev_size]
		self.neg_label = neg_label[dev_size:]
		self.dev_pos_label = pos_label[:dev_size]
		self.pos_label = pos_label[dev_size:]
		self.dev_sent_Pdist = sent_Pdist[:dev_size]
		self.sent_Pdist = sent_Pdist[dev_size:]
		self.dev_sent_Ndist = sent_Ndist[:dev_size]
		self.sent_Ndist = sent_Ndist[dev_size:]
		self.dev_polarity = polarity[:dev_size]
		self.polarity = polarity[dev_size:]
		self.aspect_idx = aspect_idx
		self.length = len(self.train_x)
		self.current = 0

		self.g_train_x = g_train
		(g_sent_dist, g_pos_label) = g_label
		self.g_sent_dist = g_sent_dist
		self.g_pos_label = g_pos_label

	def suffle_data(self, train_x, labels):
		index = np.random.permutation(np.arange(len(train_x)))
		train_x = train_x[index]
		(use_Pdist, use_Ndist, neg_label, pos_label, sent_Pdist, sent_Ndist, polarity) = labels
		use_Pdist = use_Pdist[index]
		use_Ndist = use_Ndist[index]
		neg_label = neg_label[index]
		pos_label = pos_label[index]
		sent_Pdist = sent_Pdist[index]
		sent_Ndist = sent_Ndist[index]
		polarity = polarity[index]
		labels = (use_Pdist, use_Ndist, neg_label, pos_label, sent_Pdist, sent_Ndist, polarity)

		return train_x, labels

	def next_batch(self, size):
		if self.current == 0:
			index = np.random.permutation(np.arange(len(self.train_x)))
			self.train_x = self.train_x[index]
			self.use_Pdist = self.use_Pdist[index]
			self.use_Ndist = self.use_Ndist[index]
			self.neg_label = self.neg_label[index]
			self.pos_label = self.pos_label[index]
			self.sent_Pdist = self.sent_Pdist[index]
			self.sent_Ndist = self.sent_Ndist[index]
			self.polarity = self.polarity[index]
		if self.current + size < self.length:
			x = self.train_x[self.current:self.current+size]
			u_p, u_n = self.use_Pdist[self.current:self.current+size], self.use_Ndist[self.current:self.current+size]
			p_l, n_l = self.pos_label[self.current:self.current+size], self.neg_label[self.current:self.current+size]
			s_p, s_n = self.sent_Pdist[self.current:self.current+size], self.sent_Ndist[self.current:self.current+size]
			self.current += size
		else:
			x = self.train_x[self.current:]
			u_p, u_n = self.use_Pdist[self.current:], self.use_Ndist[self.current:]
			p_l, n_l = self.pos_label[self.current:], self.neg_label[self.current:]
			s_p, s_n = self.sent_Pdist[self.current:], self.sent_Ndist[self.current:]
			self.current = 0

		return x, p_l, n_l, u_p, u_n, s_p, s_n

class VocabularyProcessor(object):
	def __init__(self, max_document_length, vocabulary, unknown_limit=float('Inf'), drop=False):
		self.max_document_length = max_document_length
		self._reverse_mapping = ['<UNK>', '<EOS>'] + vocabulary
		self.make_mapping()
		self.unknown_limit = unknown_limit
		self.drop = drop

	def make_mapping(self):
		self._mapping = {}
		for i, vocab in enumerate(self._reverse_mapping):
			self._mapping[vocab] = i

	def transform(self, raw_documents):
		data = []
		lengths = []
		for tokens in raw_documents:
			word_ids = np.ones(self.max_document_length, np.int32) * self._mapping['<EOS>']
			length = 0
			unknown = 0
			if self.drop and len(tokens.split()) > self.max_document_length:
				continue
			for idx, token in enumerate(tokens.split()):
				if idx >= self.max_document_length:
					break
				word_ids[idx] = self._mapping.get(token, 0)
				length = idx
				if word_ids[idx] == 0:
					unknown += 1
			length = length+1
			if unknown <= self.unknown_limit:
				data.append(word_ids)
				lengths.append(length)

		data = np.array(data)
		lengths = np.array(lengths)

		return data
			# yield word_ids
	def save(self, filename):
		with gfile.Open(filename, 'wb') as f:
			f.write(pickle.dumps(self))
	@classmethod
	def restore(cls, filename):
		with gfile.Open(filename, 'rb') as f:
			return pickle.loads(f.read())

def load_data(polarity_path, aspect_path, prepro_tpath, prepro_lpath, vocab_path):
	Aspects = ['服務', '環境', '價格', '交通', '餐廳']
	parser = Parser()

	use_Pdist = []
	sent_Pdist = []
	use_Ndist = []
	sent_Ndist = []
	pos_label = []
	neg_label = []
	train_sent = []
	polarity = []

	with open(polarity_path, 'r') as f:
		for line in f.readlines():
			line = line.strip().split("\t", 1)
			label = int(line[0])
			sent = line[1].strip()
			s_sent = parser.parse(sent)
			if len(s_sent) > max_seqlen:
				continue
			s_sent = ' '.join(s_sent)
			train_sent.append(s_sent)
			polarity.append(label)

			if label == 1:
				pos_label.append(1)
				neg_label.append(0)
			else:
				pos_label.append(0)
				neg_label.append(1)

			use_Pdist.append(False)
			sent_Pdist.append(np.array([0, 0, 0, 0 ,0]))

			use_Ndist.append(False)
			sent_Ndist.append(np.array([0, 0, 0, 0 ,0]))

	
	g_sent_dist = []
	g_pos_label = []
	g_train_sent = []
	with open(aspect_path, 'r') as f:
		keep = True
		sent = None
		p_label = 0
		pdist = np.array([0, 0, 0, 0 ,0])
		ndist = np.array([0, 0, 0, 0 ,0])
		for idx, line in enumerate(f.readlines()):
			if idx%4 == 1: # sent
				sent = line.strip()
				s_sent = parser.parse(sent)
				if len(s_sent) > max_seqlen:
					keep = False
				sent = ' '.join(s_sent)
			elif idx%4 == 2: # pos
				pdist = np.array([0, 0, 0, 0 ,0])
				aspects = line.strip().split()
				for aspect in aspects:
					pdist[Aspects.index(aspect)] = 1
				if sum(pdist) != 0:
					p_label = 1
				else:
					p_label = 0
			elif idx%4 == 3: # neg
				ndist = np.array([0, 0, 0, 0 ,0])
				aspects = line.strip().split()
				for aspect in aspects:
					ndist[Aspects.index(aspect)] = 1
				if sum(ndist) != 0:
					if p_label == 1:
						keep = False
				if keep:
					g_train_sent.append(sent)
					if p_label == 1:
						g_sent_dist.append(pdist)
					else:
						g_sent_dist.append(ndist)
					g_pos_label.append(p_label)

				keep = True
				p_label = 0

	# build vocab
	vocab = collections.defaultdict(int)
	vocabulary = []
	max_len = 0
	avg = 0
	for sent in train_sent:
		s_sent = sent.split()
		avg += len(s_sent)
		if len(s_sent) > max_len:
			max_len = len(s_sent)
		for w in s_sent:
			vocab[w] += 1
	for k, v in sorted(vocab.items(), key=lambda x:x[1], reverse=True):
		if v >= min_count:
			vocabulary.append(k)

	vocab_processor = VocabularyProcessor(max_document_length=max_seqlen, vocabulary=vocabulary)
	train_x = vocab_processor.transform(train_sent)

	aspect_idx = [0, 0, 0, 0, 0]
	for idx, aspect in enumerate(Aspects):
		index = vocab_processor._mapping[aspect]
		aspect_idx[idx] = index

	aspect_idx = np.array(aspect_idx)
	use_Pdist = np.array(use_Pdist)
	use_Ndist = np.array(use_Ndist)
	neg_label = np.array(neg_label)
	pos_label = np.array(pos_label)
	sent_Pdist = np.array(sent_Pdist)
	sent_Ndist = np.array(sent_Ndist)
	polarity = np.array(polarity)


	g_train_x = vocab_processor.transform(g_train_sent)
	g_sent_dist = np.array(g_sent_dist)
	g_pos_label = np.array(g_pos_label)

	print(len(g_pos_label))
	print(len(g_train_x))


	labels = (use_Pdist, use_Ndist, neg_label, pos_label, sent_Pdist, sent_Ndist, polarity, aspect_idx)
	cPickle.dump(train_x, open(prepro_tpath, 'wb'))
	cPickle.dump(labels, open(prepro_lpath, 'wb'))

	cPickle.dump(g_train_x, open("./prepro/g_train.dat", 'wb'))
	cPickle.dump((g_sent_dist, g_pos_label), open("./prepro/g_label.dat", 'wb'))

	vocab_processor.save(vocab_path)

	print(max_len)

	return train_x, labels, vocab_processor, g_train_x, (g_sent_dist, g_pos_label)

def load_new_aspect(aspect_file):
	f = open(aspect_file, 'r')
	r_words = []
	r_words_dict = {}
	aspect = ['服務', '環境', '價格', '交通', '餐廳']
	for idx, line in enumerate(f.readlines()):
		line = line.strip().split()
		r_words += line
		for asp in line:
			r_words_dict[asp] = aspect[idx]

	return r_words, r_words_dict

def get_unknown_word_vec(dim_size):
	return np.random.uniform(-0.25, 0.25, dim_size) 

def build_w2v_matrix(vocab_processor, w2v_path, vector_path, dim_size):
	w2v_dict = {}
	f = open(vector_path, 'r')
	for line in f.readlines():
		word, vec = line.strip().split(' ', 1)
		w2v_dict[word] = np.loadtxt([vec], dtype='float32')

	vocab_list = vocab_processor._reverse_mapping
	w2v_W = np.zeros(shape=(len(vocab_list), dim_size), dtype='float32')

	for i, vocab in enumerate(vocab_list):
		# unknown vocab
		if i == 0:
			continue
		else:
			if vocab in w2v_dict:
				w2v_W[i] = w2v_dict[vocab]
			else:
				w2v_W[i] = get_unknown_word_vec(dim_size)

	cPickle.dump(w2v_W, open(w2v_path, 'wb'))

	return w2v_W

def load_test(vocab_processor, test_path, len_limit=0):
	parser = Parser()
	test_dict = {}
	with open(test_path, 'r') as f:
		curr_id = None
		for idx, line in enumerate(f.readlines()):
			if idx % 2 == 0:
				curr_id = int(line.strip())
			if idx % 2 == 1:
				sent = line.strip()
				s_sent = parser.parse(sent)
				if len(s_sent) >= len_limit:
					sents = re.split(r"[,，！？。;!]", sent)
					p_sents = [' '.join(parser.parse(sent)) for sent in sents if sent != '']
				else:
					p_sents = [' '.join(parser.parse(sent))]
				test_x = vocab_processor.transform(p_sents)
				test_dict[curr_id] = {"raw_context":line.strip(),
									"parsed_sents":p_sents,
									"test_x":test_x
									}
	return test_dict








