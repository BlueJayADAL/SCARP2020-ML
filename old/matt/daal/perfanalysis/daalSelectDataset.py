import ujson as json
import sys
import gzip
from collections import defaultdict
import argparse
import os
import numpy as np
import random

def groupDataset(joyFolder):
	filesMap = defaultdict()
	files = os.listdir(joyFolder)
	allMalware = []
	for file in files:
		fileName = file[0:-9]
		fileSpec = fileName.split('_')
		g = fileSpec[0] #grade
		c = fileSpec[1] #class
		n = fileSpec[2] #number
		if g != "Benign":
			allMalware.append(fileName)	
		if g not in filesMap:
			filesMap[g] = defaultdict()
		if c not in filesMap[g]:
			filesMap[g][c] = []
		filesMap[g][c].append(fileName)
	#tmpMap = defaultdict()
	#tmpMap["Malware"] = allMalware
	#json.dump(tmpMap, open('./selection/allMalware.json', 'w'))
	return filesMap

def selectFeature(filesMap, malSelect, featureFolder, feature, outFile):
	fSelection = defaultdict()
	candList = []
	#select malware first
	if malSelect == None:
		for g in list(filesMap.keys()):
			if g == "Benign":
				continue
			for c in list(filesMap[g].keys()):
				cand = filesMap[g][c] #cand is a list of files for a malware
				candList.append(random.choice(cand))
	else:
		with open(malSelect, 'r') as fp:
			malMap = json.load(fp)
			candList = malMap["Malware"]

	#count the number of flows that are with given feature
	total = 0
	for cand in candList:
		#try:
		json_file = featureFolder + cand + "_" + feature + ".json" 
		with open(json_file, 'r') as fp:
			featureMap = json.load(fp)
			total += featureMap["total"+feature]
		#except:
		#	continue
	fSelection["Malware"] = candList
	fSelection["MalwareCount"] = total
	#iterate through the benign dataset to match the number of flows of malware
	candList = []
	for c in list(filesMap["Benign"].keys()):
		cand = filesMap["Benign"][c]
		candList += cand
	beList = []
	count = 0
	while count<total and len(candList) > 0:
		#try:
		index = random.randrange(len(candList))
		cand = candList[index]
		beList.append(cand)
		del candList[index]
		json_file = featureFolder + cand + "_" + feature + ".json" 
		with open(json_file, 'r') as fp:
			featureMap = json.load(fp)
			count += featureMap["total"+feature]
		#except:
		#	continue
	fSelection["Benign"] = beList
	fSelection["BenignCount"] = count

	#save the selection into JSON file
	print(fSelection) 
	json.dump(fSelection, open(outFile, 'w'))

def Select(inputFolder, outPath):
	#setup input folder and output folders
	#input folder
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
	#output folder
	if outPath == None:
		print("No valid output JSON file")
		return
	else:
		outFile = outPath 

#	featureCount = 0
#	if args.tls == True:
#		featureCount += 1
#	if args.dns == True:
#		featureCount += 1
#	if args.http == True:
#		featureCount += 1
#	if featureCount != 1:
#		print("Please specify only 1 feature, which is one of tls, dns or http")
#		return

	#get the file grade and class
	print("Collecting the file architecture of " + joyFolder)
	fMap = groupDataset(joyFolder)
	print("Done with file architecture")
	
	malSelect=None
#	if args.tls == True:
	TLS_JSON_Folder = parentFolder+"TLS_JSON/"
	if not os.path.exists(TLS_JSON_Folder):
		print("The TLS_JSON folder does not exist!")
		return
	selectFeature(fMap, malSelect, TLS_JSON_Folder, "TLS", outFile)

#	elif args.dns == True:
#		DNS_JSON_Folder = parentFolder+"DNS_JSON/"
#		if not os.path.exists(DNS_JSON_Folder):
#			print("The DNS_JSON folder does not exist!")
#			return
#		selectFeature(fMap, args.malSelect, DNS_JSON_Folder, "DNS", outFile)
#	elif args.http == True:
#		HTTP_JSON_Folder = parentFolder+"HTTP_JSON/"
#		if not os.path.exists(HTTP_JSON_Folder):
#			print("The HTTP_JSON folder does not exist!")
#			return
#		selectFeature(fMap, args.malSelect, HTTP_JSON_Folder, "HTTP", outFile)


