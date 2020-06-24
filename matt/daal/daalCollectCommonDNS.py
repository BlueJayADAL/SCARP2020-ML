import ujson as json
import sys
import gzip
from collections import defaultdict
import argparse
import os
import numpy as np
import urllib, re
import time

#This file is used to collect the most common DNS
#(1) suffixes
#(2) TTLs
suffixDict = defaultdict()
suffixList = []
ttlDict = defaultdict()
ttlList = []

def extractCommonDNS(dns):
	global suffixDict
	global suffixList
	global ttlDict
	global ttlList
	for key in list(dns.keys()):
		if key == "totalDNS":
			continue
		count = dns[key]['count']
		#for suffix
		suffix = dns[key]['suffix']
		if suffix in suffixDict:
			suffixDict[suffix] += count
		else:
			suffixDict[suffix] = np.uint64(count)
		#for ttl
		for ttl in dns[key]['ttls']:
			if ttl in ttlDict:
				ttlDict[ttl] += count
			else:
				ttlDict[ttl] = np.uint64(count)

def finalize():
	global suffixDict
	global suffixList
	global ttlDict
	global ttlList
	#for suffix
	for key in list(suffixDict.keys()):
		suffixList.append( (key, suffixDict[key]) )
	suffixList.sort(key=lambda x: x[1], reverse=True)
	suffixList = suffixList[0:40]
	suffixDict = defaultdict()
	for i in range(0, len(suffixList)):
		suffixDict[suffixList[i][0]] = i

	#for ttl
	for key in list(ttlDict.keys()):
		ttlList.append( (key, ttlDict[key]) )
	ttlList.sort(key=lambda x: x[1], reverse=True)
	ttlList = ttlList[0:64]
	ttlDict = defaultdict()
	for i in range(0, len(ttlList)):
		ttlDict[ttlList[i][0]] = i	
	

def main():
	global suffixDict
	global suffixList
	global ttlDict
	global ttlList
	parser = argparse.ArgumentParser(description="Collect Most Common DNS Features in Dataset", add_help=True)
	parser.add_argument('-i', '--input', action="store", help="The input folder containing DNS JSON files")
	args = parser.parse_args()

	if args.input == None or not os.path.isdir(args.input):
		print("No valid input folder!")
		return
	else:
		dnsFolder = args.input
		if not dnsFolder.endswith('/'):
			dnsFolder += '/'

	parentFolder = os.path.abspath(os.path.join(dnsFolder, os.pardir))
	if not parentFolder.endswith('/'):
		parentFolder += '/'
	DNS_COMMON_Folder = parentFolder+"DNS_COMMON/"
	if not os.path.exists(DNS_COMMON_Folder):
		os.mkdir(DNS_COMMON_Folder)
	dnsCommonFileName = DNS_COMMON_Folder + "dnsCommon.json"

	files = os.listdir(dnsFolder)
	for file in files:
		#print(file) #verbose
		with open(dnsFolder+file, 'r') as fp:
			dns = json.load(fp)
			extractCommonDNS(dns)
	finalize()

	#save 
	dnsCommon = defaultdict()
	dnsCommon['suffix'] = suffixDict
	dnsCommon['ttls'] = ttlDict
	json.dump(dnsCommon, open(dnsCommonFileName, 'w'))

if __name__ == "__main__":
	main()
