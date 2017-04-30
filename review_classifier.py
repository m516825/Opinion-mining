import sys
import numpy as np
import tensorflow as tf
import os
import data_utils
import _pickle as cPickle
from data_utils import VocabularyProcessor, Data
import progressbar as pb
from model import ReviewClassifier, ReviewClassifier2, ReviewClassifier3, ReviewClassifier4
import time
import csv

tf.flags.DEFINE_integer("embedding_size", 300, "Dimensionality of word embedding (default: 300)")
tf.flags.DEFINE_integer("hidden", 256, "hidden dimension of RNN hidden size")
tf.flags.DEFINE_integer("epoch", 20, "number of training epoch")
tf.flags.DEFINE_integer("batch_size", 100, "batch size per iteration")
tf.flags.DEFINE_integer("predict_every", 200, "predict model on dev set after this many steps (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 200)")
tf.flags.DEFINE_integer("dev_size", 200, "dev size")

tf.flags.DEFINE_float("lr", 1e-3, "training learning rate")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "drop out rate")

tf.flags.DEFINE_string("polarity_data", "./training_data/polarity_review.txt", "polarity review data path")
tf.flags.DEFINE_string("aspect_data", "./training_data/aspect_review2.txt", "aspect review data path")
tf.flags.DEFINE_string("test_data", "./training_data/test_review.txt", "test data path")
tf.flags.DEFINE_string("query", "./training_data/test.csv", "test query path")
tf.flags.DEFINE_string("vector_file", "./wiki/wiki.zh.vec", "Word representation vectors' file")
tf.flags.DEFINE_string("checkpoint_file", "", "checkpoint_file to be load")
tf.flags.DEFINE_string("w2v_data", "./prepro/w2v_W.dat", "word to vector matrix for our vocabulary")
tf.flags.DEFINE_string("prepro_train", "./prepro/train.dat", "tokenized train data's path")
tf.flags.DEFINE_string("prepro_labels", "./prepro/labels.dat", "tokenized train data's path")
tf.flags.DEFINE_string("vocab", "./vocab", "vocab processor path")
tf.flags.DEFINE_string("output", "./pred.csv", "output file")

tf.flags.DEFINE_boolean("eval", False, "Evaluate testing data")
tf.flags.DEFINE_boolean("prepro", True, "To do the preprocessing")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

r_words = ['服務', '環境', '價格', '交通', '餐廳', '態度', '人員', '客房', '設備', '空調', '房價', '地理', '位置', '早餐', '性價比']
r_words_dict = {'服務':"服務", '環境':"環境", '價格':"價格", '交通':"交通", '餐廳':"餐廳", '態度':"服務", '人員':"服務", '客房':"環境", '設備':"環境", '空調':"環境", '房價':"價格", '地理':"交通", '位置':"交通", '早餐':"餐廳", '性價比':"價格"}

class Model(object):
	def __init__(self, data, w2v_W, vocab_processor):
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = config)
		self.data = data
		self.w2v_W = w2v_W
		self.vocab_processor = vocab_processor
		self.vocab_size = len(vocab_processor._reverse_mapping)
		self.gen_path()

	def build_model(self):
		self.model = ReviewClassifier3(w2v_W=self.w2v_W, 
									vocab_size=self.vocab_size, 
									max_seq_len=len(self.data.train_x[0]), 
									hidden_size=FLAGS.hidden, 
									embedding_size=FLAGS.embedding_size, 
									aspect_idx=self.data.aspect_idx, 
									aspect_num=len(self.data.aspect_idx),
									g_size=len(self.data.g_train_x))

		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		self.optimizer = tf.train.RMSPropOptimizer(FLAGS.lr)

		tvars = tf.trainable_variables()

		grads, _ = tf.clip_by_global_norm(tf.gradients(self.model.cost, tvars), 5)

		self.updates = self.optimizer.apply_gradients(
						zip(grads, tvars), global_step=self.global_step)
		self.saver = tf.train.Saver(tf.global_variables())

	def gen_path(self):
		# Output directory for models and summaries
		timestamp = str(time.strftime('%b-%d-%Y-%H-%M-%S'))
		self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
		print ("Writing to {}\n".format(self.out_dir))
	    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)

	def train(self):
		batch_num = self.data.length//FLAGS.batch_size if self.data.length%FLAGS.batch_size==0 else self.data.length//FLAGS.batch_size + 1
		current_step = 0
		with self.sess.as_default():
			if FLAGS.checkpoint_file == "":
				self.sess.run(tf.global_variables_initializer())
			else:
				self.saver.restore(sess, FLAGS.checkpoint_file)

			for ep in range(FLAGS.epoch):
				cost = 0.
				pbar = pb.ProgressBar(widget=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=batch_num).start()
				print("Epoch {}".format(ep+1))
				for b in range(batch_num):

					x, p_l, n_l, _, _, _, _ = self.data.next_batch(FLAGS.batch_size)

					x = np.concatenate([self.data.g_train_x, x], axis=0)
					p_l = np.concatenate([self.data.g_pos_label, p_l])
					
					feed_dict = {
						self.model.seq_in:x,
						self.model.pos_label:p_l,
						self.model.neg_label:n_l,
						self.model.g_dist:self.data.g_sent_dist
						}

					loss, step, _ = self.sess.run([self.model.cost, self.global_step, self.updates], feed_dict=feed_dict)

					current_step = tf.train.global_step(self.sess, self.global_step)

					cost += loss/(batch_num + len(self.data.g_train_x))

					if current_step % 100 == 0:
						self.eval()

					pbar.update(b+1)
				pbar.finish()

				path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
				print ("\nSaved model checkpoint to {}\n".format(path))

				print (">>cost: {}".format(cost))

	def eval(self):
		feed_dict = {self.model.seq_in:self.data.dev_x}
		# pos_argmax, neg_argmax = self.sess.run([self.model.pos_argmax, self.model.neg_argmax], feed_dict=feed_dict)
		
		pos_argmax, pos_attn = self.sess.run([self.model.pos_argmax, self.model.sigmoid_attn], feed_dict=feed_dict)
		
		sentence = []
		for dev_x in self.data.dev_x:
			sent = []
			for i in dev_x:
				if self.vocab_processor._reverse_mapping[i] != '<EOS>':
					sent.append(self.vocab_processor._reverse_mapping[i])
			sentence.append(''.join(sent))

		for i in np.random.choice(FLAGS.dev_size, 10, replace=False):
			pred = 1 if pos_argmax[i] == 1 else -1
			print(sentence[i])
			print("label: {}, predict: {}".format(self.data.dev_polarity[i], pred))
			print("服務: {:.4}, 環境: {:.4}, 價格: {:.4}, 交通: {:.4}, 餐廳:{:.4}".format(pos_attn[i][0], pos_attn[i][1], pos_attn[i][2], pos_attn[i][3], pos_attn[i][4]))
			print("==================================================")

		acc = 0.
		for idx, polar in enumerate(self.data.dev_polarity):
			if polar == 1:
				acc += 1 if pos_argmax[idx] == 1 else 0
			else:
				acc += 1 if pos_argmax[idx] == 0 else 0
				# acc += 1 if neg_argmax[idx] == 1 else 0

		print("Accuarcy: {}".format(acc/FLAGS.dev_size))

def load_model(graph, sess, checkpoint_file):
	saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
	saver.restore(sess, checkpoint_file)

	seq_in = graph.get_operation_by_name("seq_in").outputs[0]
	attn_w = graph.get_operation_by_name("hops/sigmoid_attn").outputs[0]
	pred = graph.get_operation_by_name("hops/pred").outputs[0]
	# attn_w = graph.get_operation_by_name("polarity_attention/attn_w").outputs[0]
	# pred = graph.get_operation_by_name("seq_attention/pred").outputs[0]

	return {
			"seq_in":seq_in,
			"attn_w":attn_w,
			"pred":pred
	}

def main(_):

	print("Parameter:")
	for k, v in FLAGS.__flags.items():
		print("{} = {}".format(k, v))

	if not os.path.exists("./prepro"):
		os.makedirs("./prepro")

	if FLAGS.eval:
		print("Evaluation...")
		threshold = 0.5
		Aspects = ['服務', '環境', '價格', '交通', '餐廳']
		vocab_processor = VocabularyProcessor.restore(FLAGS.vocab)
		test_dict = data_utils.load_test(vocab_processor, FLAGS.test_data)
		graph = tf.Graph()
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		f = open('out.txt', 'w')
		r_words, r_words_dict = data_utils.load_new_aspect('new_aspect.txt')
		print (r_words)
		print (r_words_dict)
		with graph.as_default(), tf.Session(config = config) as sess:

			model = load_model(graph, sess, FLAGS.checkpoint_file)

			for k, v in sorted(test_dict.items(), key=lambda x:x[0]):
				attns, preds = sess.run([model["attn_w"], model["pred"]], feed_dict={model["seq_in"]:v["test_x"]})
				t_asp = {'服務':0, '環境':0, '價格':0, '交通':0, '餐廳':0}
				t_asp_s = {'服務':0, '環境':0, '價格':0, '交通':0, '餐廳':0}
				for idx, p in enumerate(preds):
					label = 1 if p == 1 else -1
					
					for word in r_words:
						if v['parsed_sents'][idx].find(word) >= 0:
							t_asp_s[r_words_dict[word]] = 1
							t_asp[r_words_dict[word]] = label
							
					# for a_idx, a in enumerate(attns[idx]):
					# 	if a >= threshold and a > t_asp_s[Aspects[a_idx]]:
					# 		t_asp_s[Aspects[a_idx]] = a 
					# 		t_asp[Aspects[a_idx]] = label
				test_dict[k]["aspect"] = t_asp
				f.write(test_dict[k]['raw_context']+'\n')
				f.write(str(t_asp))
				f.write("\n")
			
		ans = []
		with open(FLAGS.query, 'r') as f:
			for idx, row in enumerate(csv.reader(f)):
				if idx != 0:
					ans.append(test_dict[int(row[1])]["aspect"][row[2]])
		print(len(ans))
		with open(FLAGS.output, 'w') as f:
			f.write("Id,Label\n")
			for idx, p in enumerate(ans):
				f.write("{},{}\n".format(idx+1, p))

	else:
		if FLAGS.prepro:
			print("Start preprocessing data...")
			train_x, labels, vocab_processor, g_train_x, (g_sent_dist, g_pos_label) = data_utils.load_data(FLAGS.polarity_data, FLAGS.aspect_data, FLAGS.prepro_train, FLAGS.prepro_labels, FLAGS.vocab)	
			(use_Pdist, use_Ndist, neg_label, pos_label, sent_Pdist, sent_Ndist, polarity, aspect_idx) = labels
			print("Start loading pre-train word embedding...")
			w2v_W = data_utils.build_w2v_matrix(vocab_processor, FLAGS.w2v_data, FLAGS.vector_file, FLAGS.embedding_size)
		else:
			train_x = cPickle.load(open(FLAGS.prepro_train, 'rb'))
			labels = cPickle.load(open(FLAGS.prepro_labels, 'rb'))
			g_train_x = cPickle.load(open("./prepro/g_train.dat", 'rb'))
			(g_sent_dist, g_pos_label) = cPickle.load(open("./prepro/g_label.dat", 'rb'))
			(use_Pdist, use_Ndist, neg_label, pos_label, sent_Pdist, sent_Ndist, polarity, aspect_idx) = labels
			vocab_processor = VocabularyProcessor.restore(FLAGS.vocab)
			w2v_W = cPickle.load(open(FLAGS.w2v_data, 'rb'))

		print(len(vocab_processor._reverse_mapping))
		print(len(train_x))
		print(len(use_Pdist))
		print(len(w2v_W))

		data = Data(train_x, (use_Pdist, use_Ndist, neg_label, pos_label, sent_Pdist, sent_Ndist, polarity), aspect_idx, FLAGS.dev_size, g_train_x, (g_sent_dist, g_pos_label))

		model = Model(data, w2v_W, vocab_processor)

		model.build_model()

		model.train()

if __name__ == '__main__':
	tf.app.run()