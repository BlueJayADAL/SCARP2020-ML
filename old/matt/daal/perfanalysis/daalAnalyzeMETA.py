import ujson as json
import sys
import gzip
from collections import defaultdict
import argparse
import os
#import matplotlib.pyplot as plt
import numpy as np
def getMetadata(flow): 
	tmp = []
	# inbound packets
	if 'num_pkts_in' in flow:
		tmp.append(flow['num_pkts_in'])
	else:
		tmp.append(0)
	# outbound packets
	if 'num_pkts_out' in flow:
		tmp.append(flow['num_pkts_out']) 
	else:
		tmp.append(0)
	# inbound bytes
	if 'bytes_in' in flow:
		tmp.append(flow['bytes_in']) 
	else:
		tmp.append(0)
	# outbound bytes
	if 'bytes_out' in flow:
		tmp.append(flow['bytes_out']) 
	else:
		tmp.append(0)
	# elapsed time of flow
	if flow['packets'] == []:
		tmp.append(0)
	else:
		time = 0
		for packet in flow['packets']:
			time += packet['ipt']
		tmp.append(time)
	return tmp

def getTimes(flow):
	numRows = 10
	binSize = 50.0
	transMat = np.zeros((numRows,numRows))
	if len(flow['packets']) == 0:
		return list(transMat.flatten())
	elif len(flow['packets']) == 1:
		cur = min(int(flow['packets'][0]['ipt']/float(binSize)), numRows-1)
		transMat[cur, cur] = 1
		return list(transMat.flatten())
	# get raw transition counts
	for i in range(1, len(flow['packets'])):
		prev = min(int(flow['packets'][i-1]['ipt']/float(binSize)), numRows-1)
		cur = min(int(flow['packets'][i]['ipt']/float(binSize)), numRows-1)
		transMat[prev, cur] += 1	
	# get empirical transition probabilities
	for i in range(numRows):
		if float(np.sum(transMat[i:i+1])) != 0:
			transMat[i:i+1] = transMat[i:i+1]/float(np.sum(transMat[i:i+1]))
	return list(transMat.flatten())

def getLengths(flow):
	numRows = 10
	binSize = 150.0
	transMat = np.zeros((numRows,numRows))
	if len(flow['packets']) == 0:
		return list(transMat.flatten())
	elif len(flow['packets']) == 1:
		cur = min(int(flow['packets'][0]['b']/float(binSize)), numRows-1)
		transMat[cur, cur] = 1
		return list(transMat.flatten())
	# get raw transition counts
	for i in range(1, len(flow['packets'])):
		prev = min(int(flow['packets'][i-1]['b']/float(binSize)), numRows-1)
		#if 'b' not in flow['packets'][i]:
		#	break
		cur = min(int(flow['packets'][i]['b']/float(binSize)), numRows-1)
		transMat[prev, cur] += 1
	# get empirical transition probabilities
	for i in range(numRows):
		if float(np.sum(transMat[i:i+1])) != 0:
			transMat[i:i+1] = transMat[i:i+1]/float(np.sum(transMat[i:i+1]))
	return list(transMat.flatten())

def getByteDist(flow):
	if len(flow['packets']) == 0:
		return list(np.zeros(256))
	if 'byte_dist' in flow and sum(flow['byte_dist']) > 0:
		tmp = map(lambda x: x/float(sum(flow['byte_dist'])), flow['byte_dist'])
		return list(tmp)
	else:
		return list(np.zeros(256))


def ProcessMETA(inPathName, fileName, meta):
	json_file = "%s%s" % (inPathName, fileName)
	#print("processing META for %s" %(json_file)) #verbose
	#read each line and convert it into dict
	lineno = 0
	total = 0
	with gzip.open(json_file, 'r') as fp:
		for line in fp:
			lineno = lineno + 1
			try:
				tmp = json.loads(line)
			except:
				continue
			if ('version' in tmp) or ("tls" not in tmp) or (int(tmp["dp"]) != 443):
				continue
			total += 1
			serverAddr = "%s@%s@%s@%s" % (str(lineno), tmp["sa"], str(tmp["sp"]), tmp["da"])
			#print serverAddr
			try:
				meta[serverAddr]['count'] += 1
			#if serverAddr not in meta:
			except KeyError:				
				meta[serverAddr] = defaultdict()
				meta[serverAddr]['count'] = 1
				# Multithread to speed up Meta time?
				#1 times
				meta[serverAddr]['flowTimes'] = getTimes(tmp)
				#2 lengths
				meta[serverAddr]['flowLengths'] = getLengths(tmp)
				#3 byte distribution
				meta[serverAddr]['flowByteDist'] = getByteDist(tmp)
				
	try:
		meta["totalMETA"] += total	
	#if "totalMETA" not in meta:
	except KeyError:		
		meta["totalMETA"] = total
		


def saveToJson(outPathName, fileName, meta):
	fname = "%s%s_META.json" % (outPathName, (fileName.split('.'))[0])
	#print("save JSON to " + fname) #verbose
	with open(fname, 'w') as fp:
		json.dump(meta, fp)

def Analyze(inputFolder):
	#setup input folder and output folders
	if inputFolder == None or not os.path.isdir(inputFolder):
		print("No valid input folder!")
		return
	else:
		joyFolder = inputFolder
		if not joyFolder.endswith('/'):
			joyFolder += '/'
	parentFolder = os.path.abspath(os.path.join(joyFolder, os.pardir))
	if not parentFolder.endswith('/'):
		parentFolder += '/'
	META_JSON_Folder = "%sMETA_JSON/" % (parentFolder)
	if not os.path.exists(META_JSON_Folder):
		os.mkdir(META_JSON_Folder)

	files = os.listdir(joyFolder)
	for item in files:
		try:
			meta = defaultdict()
			ProcessMETA(joyFolder, item, meta) 
			saveToJson(META_JSON_Folder, item, meta)
		except:
			continue
	

