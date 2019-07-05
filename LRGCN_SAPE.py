# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import tensorflow as tf
import numpy as np
from scipy import sparse as sp
from utils import *
from path_model import LRGCN
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
import tensorflow.contrib.slim as slim
from scipy import sparse
import pickle as pkl
import networkx as nx
from load_data import load_data
import sys
from collections import defaultdict



tf.app.flags.DEFINE_integer('num_epochs', 25, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size to train in one step')
tf.app.flags.DEFINE_integer('labels', 1, 'number of label classes')
tf.app.flags.DEFINE_integer('word_pad_length', 4438, 'word pad length for training')
tf.app.flags.DEFINE_integer('decay_step', 500, 'decay steps')
tf.app.flags.DEFINE_float('learn_rate', 1e-2, 'learn rate for training optimization')
tf.app.flags.DEFINE_boolean('train', False, 'train mode FLAG')

FLAGS = tf.app.flags.FLAGS

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
tag_size = FLAGS.labels
word_pad_length = FLAGS.word_pad_length
feature_dimension = 2
lr = FLAGS.learn_rate
window_size = 24
num_path = 200
max_path_len = 49

placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(window_size)],
    'features':tf.placeholder(tf.float32, shape=(window_size,word_pad_length,feature_dimension)),
    'labels': tf.placeholder(tf.float32, shape=(num_path,)),
    'labels_mask': tf.placeholder(tf.int32,shape=(num_path,word_pad_length)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'path_node_index_array': tf.placeholder(tf.int32,shape=(num_path, max_path_len)),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

model = LRGCN()
with tf.Session() as sess:
  # build graph
  model.build_graph(n=word_pad_length,placeholders = placeholders,d =feature_dimension)
  # Downstream Application
  with tf.variable_scope('DownstreamApplication'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learn_rate = tf.train.exponential_decay(lr, global_step, FLAGS.decay_step, 0.98, staircase=True)
    labels = placeholders['labels']
    Mask = placeholders['labels_mask']
    initializer = tf.keras.initializers.he_normal()
    print("---\n model.H2 shape: {}\n----".format(model.H2.shape))
    zero_padding = tf.constant(0.0, shape=[1, 8])
    rnn_input = tf.nn.embedding_lookup(tf.concat([model.H2, zero_padding], 0), placeholders['path_node_index_array'])
    sub_w11 = tf.get_variable(name='sub_w11',shape=(32,8),initializer=initializer)
    sub_w22 = tf.get_variable(name='sub_w22',shape=(8,32),initializer=initializer)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(8)
    outputs,last_states=   tf.nn.dynamic_rnn(cell = lstm_cell,inputs = rnn_input, dtype = tf.float32)
    print("---\n outputs shape: {}\n----".format(outputs.shape))
    outputs_trans = tf.transpose(outputs, perm=[0, 2, 1])
    print("---\n outputs_trans shape: {}\n----".format(outputs_trans.shape))    
    sub_w11_stack = tf.tile(tf.expand_dims(sub_w11, 0), [num_path, 1, 1])
    sub_w22_stack = tf.tile(tf.expand_dims(sub_w22, 0), [num_path, 1, 1])
    attention_path = tf.nn.softmax(tf.matmul(sub_w22_stack, tf.tanh(tf.matmul(sub_w11_stack, outputs_trans))))
    print("---\n attention_path shape: {}\n----".format(attention_path.shape))

    path_output = tf.matmul(attention_path, outputs)
    print("---\n path_output shape: {}\n----".format(path_output.shape))
    path_output = tf.reshape(path_output, [num_path, 1, 8*8])
    print("---\n path_output shape: {}\n----".format(path_output.shape))
    initializer = tf.keras.initializers.he_normal()
    fc_weights = tf.get_variable(name='fc_weights',shape=(64,1),initializer=initializer)
    fc_weights_stack = tf.tile(tf.expand_dims(fc_weights, 0), [num_path, 1, 1])
    logits = tf.reshape(tf.matmul(path_output,fc_weights_stack),[-1])
    print("---\n logits shape: {}\n----".format(logits.shape))
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits,pos_weight = 3.) 
    loss = tf.reduce_mean(loss)
    params = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learn_rate)
    grad_and_vars = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(grad_and_vars, 1)
    opt = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
    print("HERE3")
  
  def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

  sess.run(tf.global_variables_initializer())

  train_tuopu_input,train_word_input,test_tuopu_input,test_word_input,ally,ty,whole_mask, path_node_index_array = load_data(window_size) 
  idx = np.random.RandomState(seed=42).permutation(len(ally))
  train_word_input = list(map(train_word_input.__getitem__, idx))
  train_tuopu_input = list(map(train_tuopu_input.__getitem__, idx))
  ally = list(map(ally.__getitem__, idx))
  validation_size = 200
  vtrain_word_input = train_word_input[-validation_size:]
  vtrain_tuopu_input = train_tuopu_input[-validation_size:]
  vally = ally[-validation_size:]
  train_word_input = train_word_input[:-validation_size]
  train_tuopu_input = train_tuopu_input[:-validation_size]
  ally = ally[:-validation_size]
  step_print = 500
  total = len(train_word_input)
  vtotal = len(vtrain_word_input)


  for i in range(int(total)):
     train_word_input[i] = train_word_input[i].reshape((-1,train_word_input[i].shape[1]*train_word_input[i].shape[2]))
  for i in range(int(vtotal)):
     vtrain_word_input[i] = vtrain_word_input[i].reshape((-1,vtrain_word_input[i].shape[1]*vtrain_word_input[i].shape[2]))

  if FLAGS.train == True:
    print('start training')

    time1 = time.time() # for time elapsed

    hard_example = set()
    hist_loss = []
    stop_sign = 0
    for epoch_num in range(num_epochs):
      epoch_loss = 0
      step_loss = 0
      
      for i in range(int(total)):
        batch_input,batch_tuopu, batch_tags = (sp.csr_matrix(train_word_input[i] + 1),train_tuopu_input[i], np.array([1 if element>0 else 0 for element in np.sum(ally[i],axis=0)]))
        batch_input = preprocess_features(batch_input.tolil())
        batch_input = batch_input.todense()
        batch_input = np.array(batch_input).reshape(window_size,word_pad_length,2)
        batch_tuopu = [preprocess_adj(batch_tuopu[ii]) for ii  in range(len(batch_tuopu))]
        train_ops = [opt, loss, learn_rate, global_step]
        
        feed_dict = construct_feed_dict(batch_input, batch_tuopu, batch_tags,whole_mask, placeholders)
        feed_dict.update({placeholders['path_node_index_array']: path_node_index_array})
        result = sess.run(train_ops, feed_dict=feed_dict)
        step_loss += result[1]
        epoch_loss += result[1]
        if epoch_num == num_epochs -1 and result[1] > 1.3:
          hard_example.add(i)
        if i % step_print == (step_print-step_print):          
          print("step_log: (epoch:", '%04d' % (epoch_num), "step:", '%04d' % (i), "global_step:", '%04d' % (result[3]), "learn_rate:", "{:.5f}".format(result[2]), "Loss:", "{:.5f}".format(step_loss/step_print))
          step_loss = 0
      print('***')
      print("epoch ",'%04d' % (epoch_num),": global_step:","{:.5f}".format(result[3]), "Average Loss:", "{:.5f}".format(epoch_loss/(total/batch_size)))
      print('***\n')
      vepoch_loss = 0
      for i in range(vtotal):
        batch_input,batch_tuopu, batch_tags = (sp.csr_matrix(vtrain_word_input[i] + 1),vtrain_tuopu_input[i],np.array([1 if element>0 else 0 for element in np.sum(vally[i],axis=0)]))
        batch_input = preprocess_features(batch_input.tolil())
        batch_tuopu = [preprocess_adj(batch_tuopu[i]) for i  in range(len(batch_tuopu))]
        batch_input = batch_input.todense()
        batch_input = np.array(batch_input).reshape(window_size,word_pad_length,2)
        feed_dict = construct_feed_dict(batch_input, batch_tuopu, batch_tags,whole_mask, placeholders)
        feed_dict.update({placeholders['path_node_index_array']: path_node_index_array})
        feed_dict.update({placeholders['dropout']: 0})
        result = sess.run(loss, feed_dict=feed_dict)
        vepoch_loss += result
      print('***')
      print(vepoch_loss/validation_size)
      print('***\n')
      hist_loss.append(vepoch_loss/validation_size)
      if epoch_num == 3:
        saver = tf.train.Saver()
        saver.save(sess, "./model.ckpt")
        stop_sign = hist_loss[-1]
      if epoch_num > 3 and hist_loss[-1] - stop_sign < 0.:
        saver = tf.train.Saver()
        saver.save(sess, "./model.ckpt")
        stop_sign = hist_loss[-1]
      
    print(len(hard_example))
    time2 = time.time()
    print('---------------')
    print("seconds used for training ({} epochs): {}".format(num_epochs, time2 - time1))
    print('---------------')
  else:
    saver = tf.train.Saver()
    saver.restore(sess, "./pretrained/model.ckpt") # load the pretrained model
  
  
  total = len(test_word_input)
  for i in range(int(total)):
     test_word_input[i] = test_word_input[i].reshape((-1,test_word_input[i].shape[1]*test_word_input[i].shape[2]))
  RESULT = []
  RESULT_em = []
  print('start testing')
  time3 = time.time()
  for i in range(total):
    print("i: {}".format(i))
    batch_input,batch_tuopu, batch_tags = (sp.csr_matrix(test_word_input[i] + 1),test_tuopu_input[i], np.array([1 if element>0 else 0 for element in np.sum(ty[i],axis=0)]))
    batch_input = preprocess_features(batch_input.tolil())
    batch_tuopu = [preprocess_adj(batch_tuopu[ii]) for ii  in range(len(batch_tuopu))]
    batch_input = batch_input.todense()
    batch_input = np.array(batch_input).reshape(window_size,word_pad_length,2)
    feed_dict = construct_feed_dict(batch_input, batch_tuopu, batch_tags,whole_mask, placeholders)
    feed_dict.update({placeholders['path_node_index_array']: path_node_index_array})
    feed_dict.update({placeholders['dropout']: 0})
    result = sess.run([tf.nn.sigmoid(logits)], feed_dict=feed_dict)
    RESULT.append(result[0])
  prediction = np.asarray(RESULT)
  print(prediction.shape)
  y_test = np.asarray([[1 if element>0 else 0 for element in np.sum(ty[i],axis=0)] for i in range(len(ty))])
  predictions = np.asarray([[0 if j<0.5 else 1 for j in i] for i in prediction.tolist()]).astype(int)
  correct_prediction = np.equal(predictions, y_test)
  print(np.sum(correct_prediction)/float(prediction.shape[0]*prediction.shape[1]))
  print(confusion_matrix(y_test.reshape([-1]),predictions.reshape([-1])))
  precision, recall, fscore, support = score(y_test.reshape([-1]), predictions.reshape([-1]))
  print('precision: {}'.format(precision))
  print('recall: {}'.format(recall))
  print('fscore: {}'.format(fscore))
  print("auc is, ",roc_auc_score(y_test.reshape([-1]),prediction.reshape([-1])))
  print('Macro-F1: {}'.format(np.average(fscore)))
  time4 = time.time()
  print('---------------')
  print("seconds used for testing: {}".format(time4 - time3))
  print('---------------')
  sess.close()
