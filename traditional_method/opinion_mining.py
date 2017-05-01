# -*- coding: utf-8 -*-
import jieba
import jieba.posseg
from gensim.models import word2vec
import sys
from collections import defaultdict
import csv
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import operator


training_data = 'polarity_review.txt'
corpus = 'corpus.txt'
# aspect_path = 'new_aspect.txt'
aspect_path = 'aspect_term.txt'
new_aspect_path = 'new_aspect.txt'
# new_aspect_path = 'best_aspect.txt'
embedding_size = 50
positive_path = 'NTUSD_pos.txt'
negative_path = 'NTUSD_neg.txt'
window_size = 4
test_file = 'test_review.txt'
pos_file = 'pos_review.txt'
neg_file = 'neg_review.txt'
my_pos = 'pos_word.txt'
my_neg = 'neg_word.txt'
stopword_file = 'c_stopword.txt'

ans_file = 'ans.txt'
test_csv = 'test.csv'
pred_file = 'pred.csv'

pos = 'cleanpos.txt'
neg = 'cleanneg.txt'

def sentence_segmentation(input_file, output_file, pos_output, neg_output):
	# sentence segmentation

	f = open(input_file,'r')
	sentences = []

	positive = []
	negative = []
	for line in f.readlines():
		sen = line.strip().split('\t')
		polarity = int(sen[0])
		sen = jieba.cut(sen[1])
		sen = ' '.join(s for s in sen)

		if polarity == 1:
			positive.append(sen)
		elif polarity == -1:
			negative.append(sen)
		sentences.append(sen)


	f2 = open(output_file,'w')
	f_pos = open(pos_output,'w')
	f_neg = open(neg_output,'w')
	for sen in sentences:
		f2.write(sen.encode('utf-8')+'\n')
	for sen in positive:
		f_pos.write(sen.encode('utf-8')+'\n')
	for sen in negative:
		f_neg.write(sen.encode('utf-8')+'\n')

def word_embedding(training_corpus):
	#  training word embedding
	sentences = word2vec.Text8Corpus(training_corpus)

	print 'load sentences'
	model = word2vec.Word2Vec(sentences,size=embedding_size, window = 8, workers=4, min_count=4,iter=50)

	model.save('model')

	return model


def add_aspect(aspect_path, new_aspect_path, model):
	# add aspect words and build aspect dictionary

	f_aspect = open(aspect_path,'r')
	aspect = []
	aspect_dict = {}
	new_aspect = []

	for idx,line in enumerate(f_aspect.readlines()):
		line = line.strip().replace('\t',' ').split(' ')
	
		aspect.append([])
		aspect_dict[line[0]] = idx
		new_aspect.append([])
		for word in line:
			aspect[idx].append(word)
			new_aspect[idx].append(word.decode('utf-8'))
	
	for idx,word_list in enumerate(aspect):
		for idx2,word in enumerate(word_list):
			word = word.decode('utf-8')

			words = []
			try:
				if idx2 == 0:
					words = model.most_similar(word,topn = 10)
				else:
					words = model.most_similar(word,topn = 5)
			except:
				print 'cannot find word ',word.decode('utf-8')
				continue
		
			new_aspect[idx].extend(w for w,p in words)
		
	f_new_aspect = open(new_aspect_path,'w')
	for words in new_aspect:
		words = set(words)
		for w in words:
			f_new_aspect.write(w.encode('utf-8')+' ')
		f_new_aspect.write('\n')


	return aspect_dict


def load_polarity_term(positive_file, negative_file):

	fp = open(positive_file,'r')
	fn = open(negative_file,'r')

	positive = []
	negative = []
	for line in fp.readlines():
		line = line.strip()
		positive.append(line)

	for line in fn.readlines():
		line = line.strip()
		negative.append(line)

	return positive, negative



def load_aspect(aspect_path):

	f = open(aspect_path,'r')
	aspect = []
	for idx,line in enumerate(f.readlines()):
		line = line.strip().split(' ')
		aspect.append([])
		aspect[idx].extend(line)

	return aspect


def test(test_file, positive, negative, aspect_dict, new_aspect, window_size, output_file, test_csv, pred_file):


	f = open(test_file,'r')
	review_id = []
	review_id_dic = {}
	review_sen = []
	need_pred = []
	for idx, line in enumerate(f.readlines()):
		line = line.strip()
		if idx%2 == 0:
			review_id.append(int(line))
			review_id_dic[int(line)] = idx/2
			need_pred.append([])
			
		else:
			review_sen.append([])
			sen = jieba.cut(line)
			review_sen[idx/2].extend(s for s in sen)


	f = open('new_test.txt','w')
	for sen in review_sen:
		string = ' '.join(s.encode('utf-8') for s in sen)
		f.write(string+'\n')


	f_out = open(pred_file,'w')
	f_test = open(test_csv,'r')

	f_out.write('Id,Label\n')

	for idx, row in enumerate(csv.reader(f_test)):

		if idx!=0:
			rid = int(row[1])
			asp = row[2]
			need_pred[review_id_dic[rid]].append(aspect_dict[asp])

	context = []
	flag = 0
	cnt = 1

	n_review = len(review_id)
	opinion = ''

	for idx, sen in enumerate(review_sen):
		print idx
		pred_aspect = [0,0,0,0,0]
		flag == 0
		for idx2, word in enumerate(sen):

			sen_len = len(sen)
			word = word.encode('utf-8')
	
			for a_idx, aspect in enumerate(new_aspect):
				if word in aspect:
					new_idx = idx2 + 1
					for i in range(window_size):
						if new_idx + i >= sen_len:
							break 
						w = sen[new_idx+i].encode('utf-8')
						if w in positive:
							pred_aspect[a_idx] = 1
							flag = 1
							opinion = w
						elif w in negative:
							pred_aspect[a_idx] = -1
							opinion = w
							flag = -1
							break
					print 'find!',review_id[idx],a_idx, word, opinion, flag

		for idx2, pred in enumerate(need_pred[idx]):
			# print 'review_id ', review_id[idx],'idx = ',idx2, ', pred:',pred, 'pred_aspect ', pred_aspect[pred]
			f_out.write(str(cnt)+','+str(pred_aspect[pred])+'\n');
			cnt+=1


		# print >> sys.stderr, str(idx), '/', str(n_review)

def polarity_word(polarity_file, output_file, stopword_file):

	f = open(polarity_file,'r')
	f_stopword = open(stopword_file,'r')
	f_out = open(output_file,'w')
	stop_word = []
	for line in f_stopword.readlines():
		line = line.strip()
		stop_word.append(line)


	polarity_dic = defaultdict(lambda:0)

	for line in f.readlines():
		line = line.strip().split(' ')
		for word in line:
			if word not in stop_word:
				polarity_dic[word] +=1


	sorted_dict = sorted(polarity_dic.items(), key=operator.itemgetter(1), reverse=True)

	cnt = 0
	for key, value in sorted_dict:
		f_out.write(key+'\n')
		print key.decode('utf-8'),value
		cnt+=1
		if cnt > 100:
			break

def clean_ntusd_word(ntusd_file,my_file,stopword_file,output_file):
	
	f_ntu = open(ntusd_file,'r')
	f_my = open(my_file,'r')
	f_stopword = open(stopword_file,'r')
	f_out = open(output_file,'w')
	word = []
	stop_word = []

	for line in f_stopword.readlines():
		line = line.strip()
		stop_word.append(line)
	for line in f_ntu.readlines():
		line = line.strip()
		if line not in stop_word:
			word.append(line)

	for line in f_my.readlines():
		line = line.strip()
		if line not in word:
			word.append(line)


	for w in word:
		f_out.write(w+'\n')


if __name__ == '__main__':
	print 'test'


	# polarity_word(pos_file,'pos_word.txt',stopword_file)
	# polarity_word(neg_file,'neg_word.txt',stopword_file)

	# clean_ntusd_word(positive_path,my_pos,stopword_file,pos)
	# clean_ntusd_word(negative_path,my_neg,stopword_file,neg)
	
	# sentence_segmentation(training_data, corpus, pos_file, neg_file)
	model = word2vec.Word2Vec.load('model')
	aspect_dict = add_aspect(aspect_path, new_aspect_path, model)

	positive, negative = load_polarity_term(pos,neg)
	
	new_aspect = load_aspect(new_aspect_path)
	test(test_file, positive, negative, aspect_dict, new_aspect, window_size, ans_file, test_csv, pred_file)



