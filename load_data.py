from scipy import sparse
import numpy as np
import networkx as nx
from scipy.sparse import block_diag
import pickle as pkl

''' Data explanation
* File 'output.pkl' contains a dictionary. 
The key 'adjacent_matrix' corresponds to the adjacency matrix of the traffic graph. 
The key 'node_name_map' maps the id of nodes to its name. Note that the node id begins at 0. 
The key 'path_dict' contains the path id and the nodes on every path.

* File 'input_feature.pkl' is a numpy.ndarray. 
Its shape is (number of timesteps, number of nodes, number of feature dimension). 
For the provided traffic data, the data is recorded per hour from June to August. So the number of timesteps is 2160. 
The number of nodes is 4438. And The number of feature dimension is 2.

* File 'path_label.pkl' is a 0/1 numpy.ndarray. Its shape is (number of timesteps, number of paths).

* File 'adj_list.pkl' is numpy.ndarray. Its size is (1 + number of timesteps, number of nodes, number of nodes). adj_list[t] corresponds to the adjacent matrix
at time t. Note that the first adjacent matrix (adj_list[0]) corresponds to the underline graph.

'''

def sample_mask(idx, l):
  mask = np.zeros(l)
  mask[idx] = 1
  return np.array(mask, dtype=np.bool)


def load_data(window_size = 24,test_size = 500):
  with open("DATA/adj_list.pkl","rb") as f:
    adj_list = pkl.load(f)
  with open("DATA/output.pkl","rb") as f:
    output = pkl.load(f)
  whole_tuopu = output['adjacent_matrix']
  path_index = output['path_dict']
  with open("DATA/input_feature.pkl","rb") as f:
    feature = pkl.load(f)
  with open("DATA/path_label.pkl","rb") as f:
    label = pkl.load(f)
  whole_mask = np.asarray([sample_mask(path_index[i],whole_tuopu.shape[0]) for i in range(len(path_index))])
  adj_list_transpose = []
  for tuopu in adj_list:
    adj_list_transpose.append(tuopu.transpose()) 
  T_timestamp = feature.shape[0]
  slides_no = T_timestamp - window_size + 1
  fea_list = []
  tuopu_list = []
  
  for i in range(slides_no):
    fea_list.append(feature[i:i+window_size,:,:])
    tuopu_list.append(adj_list_transpose[1+i:1+i+window_size])
  fea_train_list = fea_list[:-test_size]
  fea_test_list = fea_list[-test_size:]
  ad_train_list = tuopu_list[:-test_size]
  ad_test_list = tuopu_list[-test_size:]
  label_list = []
  for i in range(slides_no):
    label_list.append(label[i:i+1,:])
  label_train_list = label_list[:-test_size]
  label_test_list = label_list[-test_size:]

  num_nodes = whole_tuopu.shape[0]
  num_path = len(path_index)
  max_path_len = 0
  for i in range(num_path):
    if len(path_index[i]) > max_path_len:
      max_path_len = len(path_index[i])

  path_node_index_array = np.full((num_path, max_path_len), int(num_nodes), dtype=np.int32)
  for i in range(num_path):
    for j in range( len(path_index[i]) ):
      path_node_index_array[i, j] = path_index[i][j]
  path_node_index_array = np.asarray(path_node_index_array)
  return ad_train_list,fea_train_list,ad_test_list,fea_test_list,label_train_list,label_test_list,whole_mask, path_node_index_array

    
