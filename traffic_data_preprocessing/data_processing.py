# from __future__ import print_function 
import numpy as np
import pickle as pkl
import csv
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import spatial
import os
import math

from collections import defaultdict



def load_station_list(filename):
	node_list = []
	with open(filename, "r") as f:
		lines = f.readlines()
		line = lines[0] # the commonts
		print("{}".format(lines[0]), end='')
		line = lines[1] #
		node_list = [int(x) for x in line.strip().split(",")]

	# print(node_list)
	return node_list



def load_station_district(filename):
	# load station 
	print("load_station_district")
	station_with_district = defaultdict(list)
	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count=0
		for row in csv_reader:
			# if line_count >= 1:
			# 	break
			node_id = int(row[1])
			district = int(row[2])
			assert district==7, "ERROR not district 7"


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

			# freeway_list.append(freeway_id)
			# node_list.append(node_id)
			# latitude_list.append(latitude)
			# longitude_list.append(longitude)

			line_count += 1
	return node_with_attribute


def construct_graph(node_list, node_with_attribute):
	print("node_list length: {}".format(len(node_list)))
	print(len(node_with_attribute))
	## construct the nodes
	freeway_with_station = defaultdict(dict)
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
		else:
			freeway_with_station[freeway_id][direction] = []
			freeway_with_station[freeway_id][direction].append(node_id)

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
		G.node[node_name_to_id[node_id]]['pos'] = (longitude, latitude)

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

	return G, node_id_to_name, node_name_to_id
	

	
def generate_feature(node_list, node_name_to_id, station_profile):
	# t = '06/01/2018 00:00:00'
	# day = t.split()[0]
	# day = day.split('/')
	# print(day)
	
	begin_time = np.datetime64('2018-06-01 00:00:00')
	end_time = np.datetime64('2018-08-29 23:00:00')

	num_time_steps = (end_time - begin_time) / np.timedelta64(1,'h') + 1
	num_time_steps = int(num_time_steps)
	print("num_time_steps: {}".format(num_time_steps))
	num_nodes = len(node_list)

	num_features = 2 # speed and 
	input_feature = np.zeros((num_time_steps, num_nodes, num_features)) 
	node_feature_omits = np.zeros((num_nodes, num_features)) ## helper 
	node_feature_total = np.zeros((num_nodes, num_features))
	
	with open(station_profile) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count % 10000 == 0:
				print("process: {}".format(line_count / 3490560.0))

			node_name = int(row[1])
			if node_name not in node_name_to_id:
				line_count += 1
				continue

			day = row[0].split()[0].split("/")
			day_new = day[2] + '-' + day[0] + '-' + day[1]
			time = day_new + " " + row[0].split()[1]
			this_time = np.datetime64(time)						

			if this_time >= begin_time and this_time <= end_time:
				assert len(row) >= 12, "line_count: {}, row: {}".format(line_count, row)
				time_id = (this_time - begin_time) / np.timedelta64(1, "h")
				# http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_hour&district_id=7&submit=Submit
				# https://stackoverflow.com/questions/9573244/most-elegant-way-to-check-if-the-string-is-empty-in-python
				time_id = int(time_id)
				# node_name = int(row[1])
				samples = row[7]
				observed = row[8]
				total_flow = row[9]
				ave_occupancy = row[10]
				ave_speed = row[11]

				if node_name not in node_name_to_id:
					line_count += 1
					continue

				node_id = node_name_to_id[node_name]

				# input_feature[time_id, node_id, 0]: average speed
				if ave_speed != '':
					input_feature[time_id, node_id, 0] = float(ave_speed)
				else:
					node_feature_omits[node_id, 0] += 1
				node_feature_total[node_id, 0] += 1

				# input_feature[time_id, node_id, 1]: average occupancy
				if ave_occupancy != '':
					input_feature[time_id, node_id, 1] = float(ave_occupancy)
				else:
					node_feature_omits[node_id, 1] += 1
				node_feature_total[node_id, 1] += 1

			line_count += 1

	pkl.dump(input_feature, open("input_feature.pkl", 'wb'), protocol=2)


def generate_label(node_list, node_name_to_id, station_profile, path_list, time_offset = np.timedelta64(24, 'h')):
	print("path_list length: {}".format(len(path_list)))
	print(path_list)
	begin_time = np.datetime64('2018-06-01 00:00:00')
	end_time = np.datetime64('2018-08-29 23:00:00')
	num_time_steps = (end_time - begin_time) / np.timedelta64(1,'h') + 1
	num_time_steps = int(num_time_steps)

	num_nodes = len(node_list)
	num_paths = len(path_list)
	node_label = np.zeros((num_time_steps, num_nodes))
	path_label = np.zeros((num_time_steps, num_paths))


	map_node_to_path = defaultdict(list) #map node id to path id
	for path_id in range(len(path_list)):
		for v in path_list[path_id]:
			map_node_to_path[v].append(path_id)

	with open(station_profile) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0

		for row in csv_reader:
			if line_count % 10000 == 0:
				print("process: {}".format(line_count / 3490560.0))

			node_name = int(row[1])
			if node_name not in node_name_to_id:
				line_count += 1
				continue
			# print("row: {}, [12]: {}".format(row, row[12]))
			node_id = node_name_to_id[node_name]
			day = row[0] .split()[0].split('/')
			day_new = day[2] + '-' + day[0] + '-' + day[1]
			time = day_new + " " + row[0].split()[1]
			this_time = np.datetime64(time)

			if this_time >= begin_time + time_offset and this_time <= end_time + time_offset:
				time_id = time_id = (this_time - time_offset - begin_time) / np.timedelta64(1, "h")
				time_id = math.floor(time_id)


				if row[12] != '':
					delay35 = float(row[12])
					if delay35 > 0:
						node_label[time_id, node_id] += 1											

			line_count += 1

	# path label
	for time_id in range(num_time_steps):
		for path_id in range(num_paths):
			node_list = path_list[path_id]
			num_nodes = len(node_list)
			# two nodes
			for node1_id in range(num_nodes - 1 - 1):
				node1 = node_list[node1_id]
				node2 = node_list[node1_id + 1]
				node3 = node_list[node1_id + 2]
				if node_label[time_id, node1] > 0 and node_label[time_id, node2] > 0 and node_label[time_id, node3] > 0:
					path_label[time_id, path_id] = 1
					break;
	pkl.dump(path_label, open("path_label.pkl", 'wb'), protocol=2)

				
def sample_path(G, path_num=200):
	assert nx.is_connected(G), "not connected"
	node_list = G.nodes()
	path_list = []
	loop = 200
	count = 0
	path_len_threshold = 50
	while count < loop:
		print(count)
		[node1, node2] = np.random.choice(node_list, 2, replace=False)

		shortest_path = nx.shortest_path(G, source=node1, target=node2)
		if len(shortest_path) < path_len_threshold:
			print("len: {}".format(len(shortest_path)))
			path_list.append(shortest_path)
			count += 1
		else:
			continue

	path_dict = {}
	for i in range(len(path_list)):
		path_dict[i] = path_list[i]
	# pkl.dump(path_list, open("path_list.pkl", "wb"), protocol=2)
	pkl.dump(path_dict, open("path_dict.pkl", "wb"), protocol=2)

	return path_list, path_dict


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
	
	
	G, node_id_to_name, node_name_to_id = construct_graph(node_list, node_with_attribute)
	G_node_id = node_id_to_name.keys()
	adj_matrix = nx.adjacency_matrix(G, nodelist = sorted( G_node_id ) )
	# pkl.dump(adj_matrix, open("adj_matrix.pkl", 'wb'), protocol=2)
	# pkl.dump(G, open("G.pkl", 'wb'), protocol=2)
	# pkl.dump(node_id_to_name, open("node_id_to_name.pkl", 'wb'), protocol=2)


	station_profile = "../raw_data/d07_text_station_hour_2018_678.txt"
	generate_feature(node_list, node_name_to_id, station_profile)
	path_list, path_dict = sample_path(G, path_num=200)
	generate_label(node_list, node_name_to_id, station_profile, path_list=path_list)
	

	data = {}
	data["node_name_map"] = node_id_to_name
	G_node_id = node_id_to_name.keys()
	print(len(G_node_id))
	adj_matrix = nx.adjacency_matrix(G, nodelist = sorted( G_node_id ) )
	data["adjacent_matrix"] = adj_matrix
	data["path_dict"] = path_dict
	pkl.dump(data, open("output.pkl", "wb"), protocol=2)
	


