# from __future__ import print_function 

# generate evolving graph from closure
# remove the edges of all nodes in the range, determined by 'Full' and 'pavement'


import numpy as np
import pickle as pkl
import csv
import numpy as np
import networkx as nx
from scipy import spatial
import os
import math
import re
from collections import defaultdict


def load_station_list(filename):
	node_list = []
	with open(filename, "r") as f:
		lines = f.readlines()
		line = lines[0] # the commonts
		print("{}".format(lines[0]), end='')
		line = lines[1] #
		node_list = [int(x) for x in line.strip().split(",")]
	return node_list



def load_meta_data(filename):
	node_with_attribute = defaultdict(dict)
	node_list = []
	latitude_list = []
	longitude_list = []
	freeway_list = []
	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter='\t')
		line_count = 0
		for row in csv_reader:
			assert len(row)==18, "{}, {}".format(line_count, row)
			if line_count == 0:
				line_count += 1
				print("first line: {}".format(row))
				continue # the first line is title

			node_id = int(row[0])
			
			try:
				freeway_id = int(row[1])
				latitude = float(row[8])
				longitude = float(row[9])
				direction = row[2]
				abs_pm = float(row[7])
				node_type = row[11]
				node_name = row[13]
			except ValueError as e:
				line_count += 1
				continue
			
			if node_type == "FF":
				name_list = node_name.strip().split()
				legal_str = {'E', 'W', 'N', "S"}
				if len(name_list)==5 and str.isdigit(name_list[1]) and str.isdigit(name_list[4]) and name_list[0][0] in legal_str and name_list[3][0] in legal_str:
					None
				else:
					node_type = "FFFalse"
			
			node_with_attribute[node_id]['freeway_id'] = freeway_id 
			node_with_attribute[node_id]['latitude'] = latitude 
			node_with_attribute[node_id]['longitude']= longitude 
			node_with_attribute[node_id]['direction'] = direction
			node_with_attribute[node_id]['abs_pm'] = abs_pm
			node_with_attribute[node_id]['type'] = node_type
			node_with_attribute[node_id]['name'] = node_name

			line_count += 1
	return node_with_attribute


def construct_graph(node_list, node_with_attribute):
	print("node_list length: {}".format(len(node_list)))
	print(len(node_with_attribute))
	## construct the nodes
	freeway_with_station = defaultdict(dict)
	freeway_with_pm = defaultdict(dict)
	node_list_new = []
	node_with_pm = defaultdict(float)
	for node_id in node_list:
		print(node_id)
		if node_id not in node_with_attribute:
			continue
		node_list_new.append(node_id)
		freeway_id = node_with_attribute[node_id]['freeway_id']
		direction = node_with_attribute[node_id]['direction']
		if direction in freeway_with_station[freeway_id]:
			freeway_with_station[freeway_id][direction].append(node_id)
			freeway_with_pm[freeway_id][direction].append(node_with_attribute[node_id]['abs_pm'])
		else:
			freeway_with_station[freeway_id][direction] = []
			freeway_with_pm[freeway_id][direction] = []
			freeway_with_station[freeway_id][direction].append(node_id)
			freeway_with_pm[freeway_id][direction].append(node_with_attribute[node_id]['abs_pm'])

		abs_pm = node_with_attribute[node_id]['abs_pm']
		node_with_pm[node_id] = abs_pm
	
	node_name_to_id = defaultdict(int)
	node_id_to_name = defaultdict(int)
	for index, value in enumerate(node_list_new):
		node_name_to_id[value] = index
		node_id_to_name[index] = value

	print("len(node_list_new): {}".format(len(node_list_new)))

	G = nx.Graph()

	## the nodes on the same road
	for freeway_id in freeway_with_station:
		for direction in freeway_with_station[freeway_id]:
			node_list = freeway_with_station[freeway_id][direction]
			pm_list = []
			for node_id in node_list:
				pm = node_with_pm[node_id]
				pm_list.append(pm)
			index_list = sorted(range(len(pm_list)), key=lambda k: pm_list[k])
			node_list = np.array(node_list)
			node_list = node_list[index_list]
			freeway_with_station[freeway_id][direction] = node_list
			freeway_with_pm[freeway_id][direction] = sorted(freeway_with_pm[freeway_id][direction])
			freeway_with_pm[freeway_id][direction] = np.array(freeway_with_pm[freeway_id][direction])
			for i in range(len(node_list) - 1):
				node1 = node_list[i]
				node2 = node_list[i+1]
				node1 = node_name_to_id[node1]
				node2 = node_name_to_id[node2]
				G.add_edge(node1, node2)

	## the edge at crossing
	for node_id in node_list_new:
		latitude = node_with_attribute[node_id]['latitude']
		longitude = node_with_attribute[node_id]['longitude']
		G.node[node_name_to_id[node_id]]['pos'] = (latitude, longitude)

	freeway_with_stationPos = defaultdict(dict)
	kdTree = defaultdict(dict)
	for freeway_id in freeway_with_station:
		for direction in freeway_with_station[freeway_id]:
			freeway_with_stationPos[freeway_id][direction] = []
			node_list = freeway_with_station[freeway_id][direction]
			for node_id in node_list:
				latitude = node_with_attribute[node_id]['latitude']
				longitude = node_with_attribute[node_id]['longitude']
				freeway_with_stationPos[freeway_id][direction].append([latitude, longitude])
			kdTree[freeway_id][direction] = spatial.KDTree(freeway_with_stationPos[freeway_id][direction] )

	print("HERE")
	for node_id in node_list_new:
		node_type = node_with_attribute[node_id]['type']
		if node_type == "FF":
			node_name = node_with_attribute[node_id]['name']
			node_name_list = node_name.strip().split()
			if len(node_name_list)==5:
				print(node_name_list)
				first_freeway = int(node_name_list[1])
				second_freeway = int(node_name_list[4])
				if first_freeway in freeway_with_stationPos and second_freeway in freeway_with_stationPos:
					dir_1 = node_name_list[0][0]
					dir_2 = node_name_list[3][0]
					if dir_1 in freeway_with_stationPos[first_freeway] and dir_2 in freeway_with_stationPos[second_freeway]:
						latitude = node_with_attribute[node_id]['latitude']
						longitude = node_with_attribute[node_id]['longitude']
						pts = [latitude, longitude]
						if node_with_attribute[node_id]['freeway_id'] == first_freeway:							
							dist, ind = kdTree[second_freeway][dir_2].query(pts)
							node_2_id = freeway_with_station[second_freeway][dir_2][ind]
							# print("node_1_id: {}, ind:{}, dist:{}".format(node_id, ind, dist) )
							if dist < 0.01:
								print("node_1_id: {}, ind:{}, dist:{}".format(node_id, ind, dist) )
								G.add_edge(node_name_to_id[node_id], node_name_to_id[node_2_id])
						else:
							# print(kdTree[first_freeway][dir_1])
							dist, ind = kdTree[first_freeway][dir_1].query(pts)
							node_1_id = freeway_with_station[first_freeway][dir_1][ind]
							# print("node_2_id: {}, ind:{}, dist: {}".format(node_id, ind, dist))
							if dist < 0.01:
								print("node_2_id: {}, ind:{}, dist: {}".format(node_id, ind, dist))
								G.add_edge(node_name_to_id[node_1_id], node_name_to_id[node_id])

	return freeway_with_station, freeway_with_pm

def generate_time(input_time):
	# input_time = '06/01/2018 00:00:00'
	day = input_time.split()[0].split("/")
	day_new = day[2] + '-' + day[0] + '-' + day[1]
	time = day_new + " " + input_time.split()[1]
	this_time = np.datetime64(time)
	return this_time				

def update_adj(initial_G, input_attr, node_with_attribute, node_id_to_name, node_name_to_id):

	print("initial_G: {}".format(type(initial_G)))
	print("input_attr: {}".format(input_attr.shape))
	[num_time_step, num_nodes] = input_attr.shape
	print(num_time_step)
	nodes_list =  initial_G.nodes()
	graph_list = []
	graph_list.append(initial_G)
	adj_initial = nx.adjacency_matrix(initial_G, nodelist = sorted( nodes_list ) )
	adj_list = []
	
	 # to generate adj_list_static	
	adj_list.append(adj_initial)
	for e in initial_G.edges():
		(n1, n2) = e
		assert initial_G[n1][n2]['weight'] == 1.0
	for i in range(num_time_step):
		if i % 100 == 0:
			print("copy graph progress: {}".format(i / float(num_time_step)))
		G_new = nx.Graph(initial_G)		
		adj_new = nx.adjacency_matrix(G_new, nodelist = sorted( nodes_list ) )
		adj_list.append(adj_new)
	
	# update the adjacent matrix using closure information
	begin_time = np.datetime64('2018-06-01 00:00:00')
	end_time = np.datetime64('2018-08-29 23:00:00')
	closure_file = "../closure/2018-0678.txt"
	with open(closure_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter='\t')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
				continue
			else:
				print("closure line_count: {}".format(line_count))
				line_count += 1

				construction_type = row[13]
				work_type = row[45]

				if construction_type == "Pavement" or work_type == "Full":
					Fwy_Dir = row[3]
					if row[3] == '' or row[5]=='' or row[8] == '' or row[14] == '' or row[17] == '' :
						continue
					begin_pm = float(row[5])
					end_pm = float(row[8])
					small_pm = 0.0
					big_pm = 0.0
					if begin_pm <= end_pm:
						small_pm = begin_pm
						big_pm = end_pm
					else:
						small_pm = end_pm
						big_pm = begin_pm

					construction_begin = generate_time(row[14])
					construction_end = generate_time(row[17])
										
					Fwy_Dir = Fwy_Dir.split('-')
					fwy_id = re.findall("\d+", Fwy_Dir[0])[0]
					fwy_id = int(fwy_id)
					fwy_dir = Fwy_Dir[1]
					if fwy_id not in freeway_with_station or fwy_dir not in freeway_with_station[fwy_id]:
						continue
					pm_list = freeway_with_pm[fwy_id][fwy_dir]
					node_list_index = np.where(np.logical_and( pm_list >= small_pm, pm_list <= big_pm))
					node_list = freeway_with_station[fwy_id][fwy_dir][node_list_index]
					if len(node_list) == 0:
						continue
					print("node_list len: {}".format(len(node_list)))
					if construction_begin >= begin_time and construction_end <= end_time:
						time_id_1 = (construction_begin - begin_time) / np.timedelta64(1, "h")
						time_id_1 = math.ceil(time_id_1)
						time_id_2 = (construction_end - begin_time) / np.timedelta64(1, "h")
						time_id_2 = int(time_id_2)
						print("time span: {}".format(time_id_2 - time_id_1))
						for time_id in range(time_id_1, time_id_2 + 1):
							this_adj = adj_list[time_id]
							this_G = nx.Graph(this_adj)							
							for node_v in node_list:
								this_G.remove_node(node_name_to_id[node_v])
								this_G.add_node(node_name_to_id[node_v])

							this_adj = nx.adjacency_matrix(this_G, nodelist=sorted( nodes_list ))
							adj_list[time_id] = this_adj
				
	return adj_list


if __name__ == '__main__':
	np.random.seed(2)

	# load the node with attribute
	meta_data_file = "../meta_data/d07_text_meta_2018_05_26.txt"
	load_meta_data_flag = True
	if load_meta_data_flag:
		node_with_attribute = load_meta_data(meta_data_file)
		print("node_with_attribute len: {}".format(len(node_with_attribute)))


	# load the node list
	node_id_file = '../node_id/graph_sensor_ids_D7_filtered2.txt'
	node_list = load_station_list(filename=node_id_file)
	print("node_list len: {}".format(len(node_list)))
		
	data = pkl.load(open("./output.pkl", 'rb'))
	adj = data['adjacent_matrix']
	
	G_tmp = nx.Graph(adj)
	print(type(G_tmp))
	G = nx.Graph()

	for e in G_tmp.edges():
		(n1, n2) = e
		G.add_edge(n1, n2, weight=1.0) 
	
	station_profile = "../raw_data/d07_text_station_hour_2018_678.txt"
	node_id_to_name  = data['node_name_map']
	node_name_to_id = {}
	for index in node_id_to_name:
		this_name = node_id_to_name[index]
		node_name_to_id[this_name] = index

	freeway_with_station, freeway_with_pm = construct_graph(node_list, node_with_attribute)
	node_attr = pkl.load(open("node_attr.pkl", "rb"))
	adj_list = update_adj(initial_G=G, input_attr=node_attr, node_with_attribute=node_with_attribute, node_id_to_name=node_id_to_name, node_name_to_id=node_name_to_id)
	pkl.dump(adj_list, open("adj_list10.pkl", 'wb'), protocol=2)



