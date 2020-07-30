import ujson as json
import sys
import gzip
from collections import defaultdict
import argparse
import os
import numpy as np
import urllib, re
import time

#This file is used to collect the most common feature HTTP
nonPresenceField = ["content-type", "user-agent", "accept-language", "server", "code"]

def extractCommonHTTP(http, httpCommon):
	if "totalHTTP" in list(http.keys()):
		field = "totalHTTP"
		if field not in httpCommon:
			httpCommon[field] = np.uint64(http[field])
		else:
			httpCommon[field] += np.uint64(http[field])
				
	if '0' in list(http.keys()):
		http = http['0']
	else:		
		return	

	for field in list(http.keys()):
		if field in nonPresenceField:
			if field not in httpCommon:
				httpCommon[field] = defaultdict()
			for key in list(http[field].keys()):
				if key not in httpCommon[field]:
					httpCommon[field][key] = np.uint64(http[field][key])
				else:
					httpCommon[field][key] += np.uint64(http[field][key])
		else:
			if field not in httpCommon:
				httpCommon[field] = np.uint64(http[field])
			else:
				httpCommon[field] += np.uint64(http[field])


def finalize(httpCommon):
	threshHold = 0.01
	finalHTTP = defaultdict()

	#for non-presence
	presenceCount = defaultdict()
	for field in nonPresenceField:
		if field not in httpCommon:
			continue 
		fieldList = []
		for key in httpCommon[field]:
			fieldList.append( (key, httpCommon[field][key]) )
		fieldList.sort(key=lambda x: x[1], reverse=True)
		presenceCount[field] = sum([x[1] for x in fieldList])
		while (len(fieldList) > 0) and (float(fieldList[-1][1])/presenceCount[field] < threshHold):
			tmp = fieldList.pop()
			#print("Remove field " + str(tmp[0]) + " with probability " + str(float(tmp[1])/presenceCount[field]) + " from " + field) #verbose
		fieldDict = defaultdict()
		for i in range(0, len(fieldList)):
			fieldDict[fieldList[i][0]] = i
		finalHTTP[field] = fieldDict

	#for presence
	fieldList = []
	for field in httpCommon:
		if field == "totalHTTP":
			continue
		if field in nonPresenceField:
			fieldList.append( (field, float(presenceCount[field])/httpCommon["totalHTTP"]) )
		else:
			fieldList.append( (field, float(httpCommon[field])/httpCommon["totalHTTP"]) )
	fieldList.sort(key=lambda x: x[1], reverse=True)
	while len(fieldList) > 0 and fieldList[-1][1] < threshHold:
		tmp = fieldList.pop()
		#print("Remove field " + str(tmp[0]) + " with probability " + str(tmp[1]) + " from presence") #verbose
	fieldDict = defaultdict()
	for i in range(0, len(fieldList)):
		fieldDict[fieldList[i][0]] = i
	finalHTTP["presence"] = fieldDict
	return finalHTTP


def Collect(inputFolder):

	if inputFolder == None or not os.path.isdir(inputFolder):
		print("No valid input folder!")
		return
	else:
		httpFolder = inputFolder
		if not httpFolder.endswith('/'):
			httpFolder += '/'

	parentFolder = os.path.abspath(os.path.join(httpFolder, os.pardir))
	if not parentFolder.endswith('/'):
		parentFolder += '/'
	HTTP_COMMON_Folder = parentFolder+"HTTP_COMMON/"
	if not os.path.exists(HTTP_COMMON_Folder):
		os.mkdir(HTTP_COMMON_Folder)
	httpCommonFileName = HTTP_COMMON_Folder + "httpCommon.json"

	httpCommon = defaultdict()
#	if args.select == None:
	files = os.listdir(httpFolder)
#	else:
#		with open(args.select, 'r') as fp:
#			selMap = json.load(fp)
#			candList = selMap["Benign"] + selMap["Malware"]
#			files = [x+"_HTTP.json" for x in candList]
	for file in files:
		#print(file) #verbose
		with open(httpFolder+file, 'r') as fp:
			http = json.load(fp)
			extractCommonHTTP(http, httpCommon)
	finalHTTP = finalize(httpCommon)
	json.dump(finalHTTP, open(httpCommonFileName, 'w'))


