import ujson as json
import sys
import gzip
from collections import defaultdict
import argparse
import os
import numpy as np
import urllib, re
import time

#This file is used to collect the most common feature TLS
#(1) client ciphersuite 
#(2) client extensions
#(3) server ciphersuite 
#(4) server extensions
cCsDict = defaultdict()
cCsList = []
cExtDict = defaultdict()
cExtList = []
sCsDict = defaultdict()
sCsList = []
sExtDict = defaultdict()
sExtList = []

def extractCommonTLS(tls):
	global cCsDict
	global cCsList
	global cExtDict
	global cExtList
	global sCsDict
	global sCsList
	global sExtDict
	global sExtList

	for key in list(tls.keys()):
		if key == "totalTLS":
			continue
		#for client ciphersuite
		for item in tls[key]['clientCS']:
			if item in cCsDict:
				cCsDict[item] += 1
			else:
				cCsDict[item] = np.uint64(1)

		#for client extensions
		for item in tls[key]['clientExt']:
			if item in cExtDict:
				cExtDict[item] += 1
			else:
				cExtDict[item] = np.uint64(1)

		#for server ciphersuite
		for item in tls[key]['serverCS']:
			if item in sCsDict:
				sCsDict[item] += 1
			else:
				sCsDict[item] = np.uint64(1)

		#for server extensions
		for item in tls[key]['serverExt']:
			if item in sExtDict:
				sExtDict[item] += 1
			else:
				sExtDict[item] = np.uint64(1)

def finalize():
	#for client ciphersuite
	for key in list(cCsDict.keys()):
		cCsList.append( (key, cCsDict[key]) )
	cCsList.sort(key=lambda x: x[1], reverse=True)
	for i in range(0, len(cCsList)):
		cCsDict[cCsList[i][0]] = i

	#for client extensions
	for key in list(cExtDict.keys()):
		cExtList.append( (key, cExtDict[key]) )
	cExtList.sort(key=lambda x: x[1], reverse=True)
	for i in range(0, len(cExtList)):
		cExtDict[cExtList[i][0]] = i	
	
	#for server ciphersuite
	for key in list(sCsDict.keys()):
		sCsList.append( (key, sCsDict[key]) )
	sCsList.sort(key=lambda x: x[1], reverse=True)
	for i in range(0, len(sCsList)):
		sCsDict[sCsList[i][0]] = i

	#for server extensions
	for key in list(sExtDict.keys()):
		sExtList.append( (key, sExtDict[key]) )
	sExtList.sort(key=lambda x: x[1], reverse=True)
	for i in range(0, len(sExtList)):
		sExtDict[sExtList[i][0]] = i

def main():
	global cCsDict
	global cCsList
	global cExtDict
	global cExtList
	global sCsDict
	global sCsList
	global sExtDict
	global sExtList
	parser = argparse.ArgumentParser(description="Collect Most Common TLS Features in Dataset", add_help=True)
	parser.add_argument('-i', '--input', action="store", help="The input folder containing TLS JSON files")
	args = parser.parse_args()

	if args.input == None or not os.path.isdir(args.input):
		print("No valid input folder!")
		return
	else:
		tlsFolder = args.input
		if not tlsFolder.endswith('/'):
			tlsFolder += '/'

	parentFolder = os.path.abspath(os.path.join(tlsFolder, os.pardir))
	if not parentFolder.endswith('/'):
		parentFolder += '/'
	TLS_COMMON_Folder = parentFolder+"TLS_COMMON/"
	if not os.path.exists(TLS_COMMON_Folder):
		os.mkdir(TLS_COMMON_Folder)
	tlsCommonFileName = TLS_COMMON_Folder + "tlsCommon.json"

	files = os.listdir(tlsFolder)
	for file in files:
		#print(file) #verbose
		with open(tlsFolder+file, 'r') as fp:
			tls = json.load(fp)
			extractCommonTLS(tls)
	finalize()	

	#save 
	tlsCommon = defaultdict()
	tlsCommon['clientCS'] = cCsDict
	tlsCommon['clientExt'] = cExtDict
	tlsCommon['serverCS'] = sCsDict
	tlsCommon['serverExt'] = sExtDict
	json.dump(tlsCommon, open(tlsCommonFileName, 'w'))

if __name__ == "__main__":
	main()
