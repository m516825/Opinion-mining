import tensorflow as tf
import math
from tensorflow.contrib.layers.python.layers import layers

class ReviewClassifier(object):
	def __init__(self, w2v_W, vocab_size, max_seq_len, hidden_size, embedding_size, aspect_idx, aspect_num, g_size):

		self.seq_in = tf.placeholder(tf.int32, [None, max_seq_len], name="seq_in")
		self.pos_label = tf.placeholder(tf.int32, [None], name="pos_label")
		self.neg_label = tf.placeholder(tf.int32, [None], name="neg_label")

		self.g_dist = tf.placeholder(tf.int32, [None, 5], name="g_dist")

		with tf.device("/cpu:0"):
			if w2v_W == None:
				self.W = tf.Variable(tf.truncated_normal([vocab_size, embedding_size]), name="W")
			else:
				self.W = tf.Variable(w2v_W, name="W")

			self.seq_em = tf.nn.embedding_lookup(self.W, self.seq_in)
			self.aspect_em = tf.nn.embedding_lookup(self.W, aspect_idx) # (5, embedding_size)
		
		with tf.name_scope("info_encoder"):

			self.pos_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)
			self.f_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)
			self.b_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)

			pos_init_state = self.pos_info_cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32) #batch_size
			f_init_state = self.f_info_cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32) #batch_size
			b_init_state = self.b_info_cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32) #batch_size

			self.pos_outputs, self.pos_states = tf.nn.dynamic_rnn(self.pos_info_cell, 
														self.seq_em,  
														sequence_length=tf.fill([tf.shape(self.seq_in)[0]], max_seq_len), 
														dtype=tf.float32, 
														initial_state=pos_init_state, 
														scope='pos_info_encoder')

			self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(self.f_info_cell, 
														self.b_info_cell, 
														self.seq_em,  
														sequence_length=tf.fill([tf.shape(self.seq_in)[0]], max_seq_len), 
														dtype=tf.float32, 
														initial_state_fw=self.pos_states, 
														initial_state_bw=self.pos_states,
														scope='info_encoder')

			self.pos_info = self.pos_outputs[:, -1]
			self.outputs = tf.concat(self.outputs, axis=-1)

		with tf.variable_scope("polarity_attention") as scope:

			self.hidden_w = tf.get_variable("attnW_h", [hidden_size//2, hidden_size//2], dtype=tf.float32)
			self.query_w = tf.get_variable("attnW_q", [embedding_size, hidden_size//2], dtype=tf.float32)
			self.score_v = tf.get_variable("attnV", [hidden_size//2, 1], dtype=tf.float32)

			def attention_score_fn(polarity_info, query):
				h_out = tf.matmul(polarity_info, self.hidden_w)
				q_out = tf.matmul([query], self.query_w)
				q_out = tf.reshape(q_out, [1, hidden_size//2])
				scores = tf.matmul(tf.nn.tanh(h_out + q_out), self.score_v)

				return scores

			pos_score = []
			for asp in range(aspect_num):
				if asp > 0: scope.reuse_variables()
				p_scores = attention_score_fn(self.pos_info, self.aspect_em[asp])

				pos_score.append(p_scores)

			pos_score = tf.concat(pos_score, axis=1)
			self.g_logit = pos_score[:g_size, :]

			self.pos_attn_w = tf.nn.softmax(pos_score, name="attn_w") # (?, 5)
			self.g_attn_w = self.pos_attn_w[:g_size, :]

		with tf.variable_scope("seq_attention") as scope:
			# output_fn = lambda x:layers.linear(x, 2, biases_initializer=tf.constant_initializer(0), scope=scope)

			self.hidden_w2 = tf.get_variable("attnW_h2", [hidden_size, hidden_size], dtype=tf.float32)
			self.query_w2 = tf.get_variable("attnW_q2", [embedding_size, hidden_size], dtype=tf.float32)
			self.score_v2 = tf.get_variable("attnV2", [hidden_size, 1], dtype=tf.float32)

			self.w_out = tf.get_variable("out_W", [hidden_size, 2], dtype=tf.float32)
			self.b_out = tf.get_variable("out_b", [2], initializer=tf.constant_initializer(0), dtype=tf.float32)

			def output_fn(x):
				return tf.matmul(x, self.w_out) + self.b_out

			def attention_construct_fn(query, seq_hidden): # (300,) (?, max_seq_len, 256)
				h_out = tf.matmul(tf.reshape(seq_hidden, [-1, hidden_size]), self.hidden_w2) # (?*max_seq_len, 256)
				q_out = tf.matmul([query], self.query_w2)
				q_out = tf.reshape(q_out, [1, hidden_size])
				scores = tf.matmul(tf.nn.tanh(h_out + q_out), self.score_v2) # (?*max_seq_len, 1)
				scores = tf.reshape(scores, [-1, max_seq_len]) # (?, max_seq_len)
				attn_w = tf.expand_dims(tf.nn.softmax(scores), axis=-1) # (?, max_seq_len, 1)
				context_vector = tf.multiply(seq_hidden, attn_w) # # (?, max_seq_len, hidden_size)
				context_vector = tf.reduce_sum(context_vector, axis=1)
				
				return context_vector


			context_vectors = []
			for asp in range(aspect_num):
				if asp > 0: scope.reuse_variables()
				context_vector = attention_construct_fn(self.aspect_em[asp], self.outputs)
				context_vectors.append(context_vector)

			context_vectors = tf.transpose(tf.stack(context_vectors), [1, 0, 2])

			pos_w_reprsenation = tf.multiply(context_vectors, tf.expand_dims(self.pos_attn_w, axis=-1))

			self.pos_reprsenation = tf.reduce_sum(pos_w_reprsenation, axis=1)

			self.pos_logit = output_fn(self.pos_reprsenation)

			self.pos_argmax = tf.argmax(self.pos_logit, axis=1, name="pred")
			# self.neg_argmax = tf.argmax(self.neg_logit, axis=1)

		with tf.name_scope("loss"):
			p_loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.pos_logit, labels=self.pos_label)
			# n_loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.neg_logit, labels=self.neg_label)
			g_loss = 0.5 * tf.losses.sigmoid_cross_entropy(self.g_dist, self.g_logit, label_smoothing=0.1)

			loss = p_loss + g_loss + 0.1 * tf.nn.l2_loss(self.w_out) #- 0.001 * tf.nn.l2_loss(self.pos_attn_w)
			# loss = p_loss + n_loss #- 0.01 * (tf.nn.l2_loss(self.pos_attn_w) + tf.nn.l2_loss(self.neg_attn_w))

			self.cost = tf.identity(loss, name='cost') 



class ReviewClassifier2(object):
	def __init__(self, w2v_W, vocab_size, max_seq_len, hidden_size, embedding_size, aspect_idx, aspect_num, g_size):

		self.seq_in = tf.placeholder(tf.int32, [None, max_seq_len], name="seq_in")
		self.pos_label = tf.placeholder(tf.int32, [None], name="pos_label")
		self.neg_label = tf.placeholder(tf.int32, [None], name="neg_label")

		self.g_dist = tf.placeholder(tf.int32, [None, 5], name="g_dist")

		with tf.device("/cpu:0"):
			if w2v_W == None:
				self.W = tf.Variable(tf.truncated_normal([vocab_size, embedding_size]), name="W")
			else:
				self.W = tf.Variable(w2v_W, name="W")

			self.seq_em = tf.nn.embedding_lookup(self.W, self.seq_in)
			self.aspect_em = tf.nn.embedding_lookup(self.W, aspect_idx) # (5, embedding_size)
		
		with tf.name_scope("info_encoder"):

			self.pos_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)
			self.f_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)
			self.b_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)

			pos_init_state = self.pos_info_cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32) #batch_size
			f_init_state = self.f_info_cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32) #batch_size
			b_init_state = self.b_info_cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32) #batch_size

			self.pos_outputs, self.pos_states = tf.nn.dynamic_rnn(self.pos_info_cell, 
														self.seq_em,  
														sequence_length=tf.fill([tf.shape(self.seq_in)[0]], max_seq_len), 
														dtype=tf.float32, 
														initial_state=pos_init_state, 
														scope='pos_info_encoder')

			self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(self.f_info_cell, 
														self.b_info_cell, 
														self.seq_em,  
														sequence_length=tf.fill([tf.shape(self.seq_in)[0]], max_seq_len), 
														dtype=tf.float32, 
														initial_state_fw=self.pos_states, 
														initial_state_bw=self.pos_states,
														scope='info_encoder')

			self.pos_info = self.pos_outputs[:, -1]
			self.outputs = tf.concat(self.outputs, axis=-1)

		with tf.variable_scope("polarity_attention") as scope:

			self.hidden_w = tf.get_variable("attnW_h", [hidden_size//2, hidden_size//2], dtype=tf.float32)
			self.query_w = tf.get_variable("attnW_q", [embedding_size, hidden_size//2], dtype=tf.float32)
			self.score_v = tf.get_variable("attnV", [hidden_size//2, 1], dtype=tf.float32)

			def attention_score_fn(polarity_info, query):
				h_out = tf.matmul(polarity_info, self.hidden_w)
				q_out = tf.matmul([query], self.query_w)
				q_out = tf.reshape(q_out, [1, hidden_size//2])
				scores = tf.matmul(tf.nn.tanh(h_out + q_out), self.score_v)

				return scores

			pos_score = []
			for asp in range(aspect_num):
				if asp > 0: scope.reuse_variables()
				p_scores = attention_score_fn(self.pos_info, self.aspect_em[asp])

				pos_score.append(p_scores)

			pos_score = tf.concat(pos_score, axis=1)
			self.g_logit = pos_score[:g_size, :]

			self.pos_attn_w = tf.nn.softmax(pos_score, name="attn_w") # (?, 5)
			self.g_attn_w = self.pos_attn_w[:g_size, :]

			self.attn_v = tf.matmul(self.pos_attn_w, self.aspect_em) # (?, embedding)

		with tf.variable_scope("seq_attention") as scope:
			# output_fn = lambda x:layers.linear(x, 2, biases_initializer=tf.constant_initializer(0), scope=scope)

			self.hidden_w2 = tf.get_variable("attnW_h2", [hidden_size, hidden_size], dtype=tf.float32)
			self.query_w2 = tf.get_variable("attnW_q2", [embedding_size, hidden_size], dtype=tf.float32)
			self.score_v2 = tf.get_variable("attnV2", [hidden_size, 1], dtype=tf.float32)

			self.w_out = tf.get_variable("out_W", [hidden_size, 2], dtype=tf.float32)
			self.b_out = tf.get_variable("out_b", [2], initializer=tf.constant_initializer(0), dtype=tf.float32)

			def output_fn(x):
				return tf.matmul(x, self.w_out) + self.b_out

			def attention_construct_fn(query, seq_hidden): # (300,) (?, max_seq_len, 256)
				h_out = tf.matmul(tf.reshape(seq_hidden, [-1, hidden_size]), self.hidden_w2) # (?*max_seq_len, 256)
				h_out = tf.reshape(h_out, [-1, max_seq_len, hidden_size]) # # (?, max_seq_len, 256)
				q_out = tf.matmul(query, self.query_w2) # (?, 256)
				q_out = tf.reshape(q_out, [-1, 1, hidden_size]) # (?, 1, 256)
				scores = tf.matmul(tf.reshape(tf.nn.tanh(h_out + q_out), [-1, hidden_size]), self.score_v2) # (?*max_seq_len, 1)
				scores = tf.reshape(scores, [-1, max_seq_len]) # (?, max_seq_len)
				attn_w = tf.expand_dims(tf.nn.softmax(scores), axis=-1) # (?, max_seq_len, 1)
				context_vector = tf.multiply(seq_hidden, attn_w) # # (?, max_seq_len, hidden_size)
				context_vector = tf.reduce_sum(context_vector, axis=1)
				
				return context_vector
			
			self.pos_reprsenation = attention_construct_fn(self.attn_v, self.outputs)

			self.pos_logit = output_fn(self.pos_reprsenation)

			self.pos_argmax = tf.argmax(self.pos_logit, axis=1, name="pred")
			# self.neg_argmax = tf.argmax(self.neg_logit, axis=1)

		with tf.name_scope("loss"):
			p_loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.pos_logit, labels=self.pos_label)
			# n_loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.neg_logit, labels=self.neg_label)
			g_loss = 0.5 * tf.losses.sigmoid_cross_entropy(self.g_dist, self.g_logit, label_smoothing=0.1)

			loss = p_loss + g_loss + 0.5 * tf.nn.l2_loss(self.w_out) #- 0.001 * tf.nn.l2_loss(self.pos_attn_w)
			# loss = p_loss + n_loss #- 0.01 * (tf.nn.l2_loss(self.pos_attn_w) + tf.nn.l2_loss(self.neg_attn_w))

			self.cost = tf.identity(loss, name='cost') 

class ReviewClassifier3(object):
	def __init__(self, w2v_W, vocab_size, max_seq_len, hidden_size, embedding_size, aspect_idx, aspect_num, g_size, hops=3):

		self.seq_in = tf.placeholder(tf.int32, [None, max_seq_len], name="seq_in")
		self.pos_label = tf.placeholder(tf.int32, [None], name="pos_label")
		self.neg_label = tf.placeholder(tf.int32, [None], name="neg_label")

		self.g_dist = tf.placeholder(tf.int32, [None, 5], name="g_dist")

		with tf.device("/cpu:0"):
			if w2v_W == None:
				self.W = tf.Variable(tf.truncated_normal([vocab_size, embedding_size]), name="W")
			else:
				self.W = tf.Variable(w2v_W, name="W")

			self.seq_em = tf.nn.embedding_lookup(self.W, self.seq_in)
			self.aspect_em = tf.nn.embedding_lookup(self.W, aspect_idx) # (5, embedding_size)
			self.aspect_em = tf.stop_gradient(self.aspect_em)
		
		with tf.name_scope("info_encoder"):

			self.f_pos_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)
			self.b_pos_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)
			self.f_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)
			self.b_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)

			f_pos_init_state = self.f_pos_info_cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32) #batch_size
			b_pos_init_state = self.b_pos_info_cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32) #batch_size

			self.pos_outputs, self.pos_states = tf.nn.bidirectional_dynamic_rnn(self.f_pos_info_cell, 
														self.b_pos_info_cell, 
														self.seq_em,  
														sequence_length=tf.fill([tf.shape(self.seq_in)[0]], max_seq_len), 
														dtype=tf.float32, 
														initial_state_fw=f_pos_init_state, 
														initial_state_bw=b_pos_init_state,
														scope='pos_info_encoder')

			self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(self.f_info_cell, 
														self.b_info_cell, 
														self.seq_em,  
														sequence_length=tf.fill([tf.shape(self.seq_in)[0]], max_seq_len), 
														dtype=tf.float32, 
														initial_state_fw=self.pos_states[0], 
														initial_state_bw=self.pos_states[1],
														scope='info_encoder')

			info_vector = tf.concat(self.pos_outputs, axis=-1)[:, -1] #self.pos_outputs[:, -1] # info vector
			self.outputs = tf.concat(self.outputs, axis=-1)

		with tf.variable_scope("hops") as scope:

			self.hidden_w = tf.get_variable("attnW_h", [hidden_size, hidden_size], dtype=tf.float32)
			self.query_w = tf.get_variable("attnW_q", [embedding_size, hidden_size], dtype=tf.float32)
			self.score_v = tf.get_variable("attnV", [hidden_size, 1], dtype=tf.float32)

			def attention_score_fn(polarity_info, query):
				h_out = tf.matmul(polarity_info, self.hidden_w)
				q_out = tf.matmul([query], self.query_w)
				q_out = tf.reshape(q_out, [1, hidden_size])
				scores = tf.matmul(tf.nn.tanh(h_out + q_out), self.score_v)

				return scores

			self.hidden_w2 = tf.get_variable("attnW_h2", [hidden_size, hidden_size], dtype=tf.float32)
			self.query_w2 = tf.get_variable("attnW_q2", [embedding_size, hidden_size], dtype=tf.float32)
			self.score_v2 = tf.get_variable("attnV2", [hidden_size, 1], dtype=tf.float32)

			self.w_out = tf.get_variable("out_W", [hidden_size, 2], dtype=tf.float32)
			self.b_out = tf.get_variable("out_b", [2], initializer=tf.constant_initializer(0), dtype=tf.float32)

			def output_fn(x):
				return tf.matmul(x, self.w_out) + self.b_out

			def attention_construct_fn(query, seq_hidden): # (300,) (?, max_seq_len, 256)
				h_out = tf.matmul(tf.reshape(seq_hidden, [-1, hidden_size]), self.hidden_w2) # (?*max_seq_len, 256)
				h_out = tf.reshape(h_out, [-1, max_seq_len, hidden_size]) # # (?, max_seq_len, 256)
				q_out = tf.matmul(query, self.query_w2) # (?, 256)
				q_out = tf.reshape(q_out, [-1, 1, hidden_size]) # (?, 1, 256)
				scores = tf.matmul(tf.reshape(tf.nn.tanh(h_out + q_out), [-1, hidden_size]), self.score_v2) # (?*max_seq_len, 1)
				scores = tf.reshape(scores, [-1, max_seq_len]) # (?, max_seq_len)
				attn_w = tf.expand_dims(tf.nn.softmax(scores), axis=-1) # (?, max_seq_len, 1)
				context_vector = tf.multiply(seq_hidden, attn_w) # # (?, max_seq_len, hidden_size)
				context_vector = tf.reduce_sum(context_vector, axis=1)
				
				return context_vector


			self.g_logits = []
			for hop in range(hops):
				if hop > 0: scope.reuse_variables()

				pos_score = []
				for asp in range(aspect_num):
					if asp > 0: scope.reuse_variables()
					p_scores = attention_score_fn(info_vector, self.aspect_em[asp])

					pos_score.append(p_scores)

				pos_score = tf.concat(pos_score, axis=1)

				self.g_logit = pos_score[:g_size, :]
				self.g_logits.append(self.g_logit)

				# self.pos_attn_w = tf.nn.softmax(pos_score, name="attn_w") # (?, 5)

				self.sigmoid_attn = tf.nn.sigmoid(pos_score, name="sigmoid_attn")

				# self.attn_v = tf.matmul(self.pos_attn_w, self.aspect_em) # (?, embedding)

				self.attn_v = tf.matmul(self.sigmoid_attn, self.aspect_em) / tf.reduce_sum(self.sigmoid_attn, axis=1, keep_dims=True)

				#######

				info_vector = attention_construct_fn(self.attn_v, self.outputs)

			self.pos_logit = output_fn(info_vector)

			self.pos_argmax = tf.argmax(self.pos_logit, axis=1, name="pred")

		with tf.name_scope("loss"):
			p_loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.pos_logit, labels=self.pos_label)
			
			g_loss = []
			for hop in range(hops):
				g_loss.append(0.5 * tf.losses.sigmoid_cross_entropy(self.g_dist, self.g_logits[hop], label_smoothing=0.1))

			loss = p_loss + tf.reduce_sum(g_loss) + 0.5 * tf.nn.l2_loss(self.w_out) 
			
			self.cost = tf.identity(loss, name='cost') 

"""
use sigmoid gate for each feature!!!!
"""
class ReviewClassifier4(object):
	def __init__(self, w2v_W, vocab_size, max_seq_len, hidden_size, embedding_size, aspect_idx, aspect_num, g_size, hops=3):

		self.seq_in = tf.placeholder(tf.int32, [None, max_seq_len], name="seq_in")
		self.pos_label = tf.placeholder(tf.int32, [None], name="pos_label")
		self.neg_label = tf.placeholder(tf.int32, [None], name="neg_label")

		self.g_dist = tf.placeholder(tf.int32, [None, 5], name="g_dist")

		with tf.device("/cpu:0"):
			if w2v_W == None:
				self.W = tf.Variable(tf.truncated_normal([vocab_size, embedding_size]), name="W")
			else:
				self.W = tf.Variable(w2v_W, name="W")

			self.seq_em = tf.nn.embedding_lookup(self.W, self.seq_in)
			self.aspect_em = tf.nn.embedding_lookup(self.W, aspect_idx) # (5, embedding_size)
			self.aspect_em = tf.stop_gradient(self.aspect_em)
		
		with tf.name_scope("info_encoder"):

			self.f_pos_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)
			self.b_pos_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)
			self.f_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)
			self.b_info_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size//2, forget_bias=1.0, state_is_tuple=True)

			f_pos_init_state = self.f_pos_info_cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32) #batch_size
			b_pos_init_state = self.b_pos_info_cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32) #batch_size

			self.pos_outputs, self.pos_states = tf.nn.bidirectional_dynamic_rnn(self.f_pos_info_cell, 
														self.b_pos_info_cell, 
														self.seq_em,  
														sequence_length=tf.fill([tf.shape(self.seq_in)[0]], max_seq_len), 
														dtype=tf.float32, 
														initial_state_fw=f_pos_init_state, 
														initial_state_bw=b_pos_init_state,
														scope='pos_info_encoder')

			self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(self.f_info_cell, 
														self.b_info_cell, 
														self.seq_em,  
														sequence_length=tf.fill([tf.shape(self.seq_in)[0]], max_seq_len), 
														dtype=tf.float32, 
														initial_state_fw=self.pos_states[0], 
														initial_state_bw=self.pos_states[1],
														scope='info_encoder')

			info_vector = tf.concat(self.pos_outputs, axis=-1)[:, -1] #self.pos_outputs[:, -1] # info vector
			self.outputs = tf.concat(self.outputs, axis=-1)

		with tf.variable_scope("hops") as scope:

			self.hidden_w = tf.get_variable("attnW_h", [hidden_size, embedding_size], dtype=tf.float32)
			self.query_w = tf.get_variable("attnW_q", [embedding_size, embedding_size], dtype=tf.float32)

			def gated_fn(polarity_info, query):
				h_out = tf.matmul(polarity_info, self.hidden_w)
				q_out = tf.matmul([query], self.query_w)
				q_out = tf.reshape(q_out, [1, embedding_size])
				g_vector = tf.multiply(tf.nn.sigmoid(h_out + q_out), [query]) # (?, hidden)

				return g_vector

			self.hidden_w2 = tf.get_variable("attnW_h2", [hidden_size, hidden_size], dtype=tf.float32)
			self.query_w2 = tf.get_variable("attnW_q2", [embedding_size, hidden_size], dtype=tf.float32)
			self.score_v2 = tf.get_variable("attnV2", [hidden_size, 1], dtype=tf.float32)

			self.w_out = tf.get_variable("out_W", [hidden_size, 2], dtype=tf.float32)
			self.b_out = tf.get_variable("out_b", [2], initializer=tf.constant_initializer(0), dtype=tf.float32)

			def output_fn(x):
				return tf.matmul(x, self.w_out) + self.b_out

			def attention_construct_fn(query, seq_hidden): # (300,) (?, max_seq_len, 256)
				h_out = tf.matmul(tf.reshape(seq_hidden, [-1, hidden_size]), self.hidden_w2) # (?*max_seq_len, 256)
				h_out = tf.reshape(h_out, [-1, max_seq_len, hidden_size]) # # (?, max_seq_len, 256)
				q_out = tf.matmul(query, self.query_w2) # (?, 256)
				q_out = tf.reshape(q_out, [-1, 1, hidden_size]) # (?, 1, 256)
				scores = tf.matmul(tf.reshape(tf.nn.tanh(h_out + q_out), [-1, hidden_size]), self.score_v2) # (?*max_seq_len, 1)
				scores = tf.reshape(scores, [-1, max_seq_len]) # (?, max_seq_len)
				attn_w = tf.expand_dims(tf.nn.softmax(scores), axis=-1) # (?, max_seq_len, 1)
				context_vector = tf.multiply(seq_hidden, attn_w) # # (?, max_seq_len, hidden_size)
				context_vector = tf.reduce_sum(context_vector, axis=1)
				
				return context_vector


			self.g_logits = []
			for hop in range(hops):
				if hop > 0: scope.reuse_variables()

				g_vectors = []
				for asp in range(aspect_num):
					if asp > 0: scope.reuse_variables()
					g_vector = gated_fn(info_vector, self.aspect_em[asp])

					g_vectors.append(g_vector) # [(?, embed_size), (), (), (), ()]

				vectors = tf.transpose(tf.stack(g_vectors), [1, 0, 2])
				
				self.attn_v = tf.reduce_sum(vectors, axis=1) # (?, 5, embed_size) -> (?, embed_size)

				c_logits = []
				for asp in range(aspect_num):
					if asp > 0: scope.reuse_variables()

					c_logits.append(tf.reduce_sum(tf.multiply(self.attn_v, [self.aspect_em[asp]]), axis=-1, keep_dims=True))

				c_logits = tf.concat(c_logits, axis=1)

				self.g_logit = c_logits[:g_size, :]
				self.g_logits.append(self.g_logit)

				self.sigmoid_attn = tf.nn.sigmoid(c_logits, name="sigmoid_attn")

				#######

				info_vector = attention_construct_fn(self.attn_v, self.outputs)

			self.pos_logit = output_fn(info_vector)

			self.pos_argmax = tf.argmax(self.pos_logit, axis=1, name="pred")

		with tf.name_scope("loss"):
			p_loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.pos_logit, labels=self.pos_label)
			
			g_loss = []
			for hop in range(hops):
				g_loss.append(0.5 * tf.losses.sigmoid_cross_entropy(self.g_dist, self.g_logits[hop], label_smoothing=0.1))

			loss = p_loss + tf.reduce_sum(g_loss) + 0.5 * tf.nn.l2_loss(self.w_out) 
			
			self.cost = tf.identity(loss, name='cost') 
