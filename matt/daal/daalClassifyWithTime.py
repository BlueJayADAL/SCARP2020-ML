import numpy as np
import random
import copy
import ujson as json
import math
import time
import os
import argparse
from collections import defaultdict, OrderedDict
from daalModel import LR, DF, SVM, ANN
from vinoANN import vinoANN
from sklearn.externals import joblib
#from modelANNGPU import ANNGPU

##global variables for binary features
#DNS
numDNSFeature = 0
numCommonDNSTTL = 0
numCommonDNSSuffix = 0
#TLS
numTLSFeature = 0
numCommonTLSClientCS = 0
numCommonTLSClientExt = 0
numCommonTLSServerCS = 0
numCommonTLSServerExt = 0
#HTTP
numHTTPFeature = 0
numCommonHTTPPresence = 0
numCommonHTTPContentType = 0
numCommonHTTPUserAgent = 0
numCommonHTTPServer = 0
numCommonHTTPCode = 0
#times, lengths, byte-distribution
numTimesFeature = 0
numLengthsFeature = 0
numDistFeature = 0
#for enabled feature
dnsDir = ""
tlsDir = ""
httpDir = ""
metaDir = ""
timesDir = ""
lengthsDir = ""
distDir = ""
enableTLS = False
#for analyze the impact of each parameters
impact = []
#for the granularity of http window
timeScale = 0
#for the filter-out of TLS flows
filterOut=[]
#for initialization of the number of META features
#must be aligned with the calculation in analyzeMETA.py
def initNumTimesFeature():
	global impact
	global numTimesFeature
	numTimesFeature = 100
	print("numTimesFeature: " + str(numTimesFeature))
	tmp = []
	for i in range(0, numTimesFeature):
		tmp.append("Times_"+str(i))
	impact += tmp
def initNumLengthsFeature():
	global impact
	global numLengthsFeature
	numLengthsFeature = 100
	print("numLengthsFeature: " + str(numLengthsFeature))
	tmp = []
	for i in range(0, numLengthsFeature):
		tmp.append("Lengths_"+str(i))
	impact += tmp
def initNumDistFeature():
	global impact
	global numDistFeature
	numDistFeature = 256
	print("numDistFeature: " + str(numDistFeature))
	tmp = []
	for i in range(0, numDistFeature):
		tmp.append("Dist_"+str(i))
	impact += tmp


#for intialization of the number of DNS features
#must be aligned with the calculation below!
def initNumDNSFeature():
	global impact
	global numDNSFeature
	# ttls + suffix + alexa + length of domain name + num of numerical characters + num of wildcard or dot + num of IPs returned
	numDNSFeature = (numCommonDNSTTL+1) + (numCommonDNSSuffix+1) + 6 + 1 + 1 + 1 + 1
	print("numDNSFeature: " + str(numDNSFeature))
	#ttls
	tmp = [""] * (numCommonDNSTTL+1)
	for key in dnsCommon["ttls"].keys():
		index = dnsCommon["ttls"][key]
		tmp[index] = "TTLS_" + key
	tmp[numCommonDNSTTL] = "TTLS_Other"
	impact += tmp
	#suffix
	tmp = [""] * (numCommonDNSSuffix+1)
	for key in dnsCommon["suffix"].keys():
		index = dnsCommon["suffix"][key]
		tmp[index] = "Suffix_" + key
	tmp[numCommonDNSSuffix] = "Suffix_Other"
	impact += tmp
	#alexa
	tmp = ["Alexa_100", "Alexa_1000", "Alexa_10000", "Alexa_100000", "Alexa_1000000", "Alexa_None"]
	impact += tmp
	#length of domain name
	tmp = ["LenDomain"]
	impact += tmp
	#num of numerical characters
	tmp = ["NumNumericDomain"]
	impact += tmp
	#num of wildcard or dot
	tmp = ["NumNonAlphaNumericDomain"]
	impact += tmp
	#num of IPs returned
	tmp = ["IPReturned"]
	impact += tmp


def processDNS(dns):	
	global numCommonDNSTTL
	global numCommonDNSSuffix
	dnsDict = defaultdict()
	for key in dns.keys():
		if key == "totalDNS":
			continue
		d = dns[key]
		ips = d['ips']
		ttls = d['ttls']
		for i in range(0, len(ips)):
			ip = ips[i]
			#1 ttls
			ttl = str(ttls[i])
			if ttl in dnsCommon["ttls"]: 
				ttlIdx = dnsCommon["ttls"][ttl]
			else:
				ttlIdx = numCommonDNSTTL
			ttlList = [0] * (numCommonDNSTTL+1)
			ttlList[ttlIdx] = 1
			#2 suffix
			suffix = d['suffix']
			if suffix in dnsCommon["suffix"]:
				suffixIdx = dnsCommon["suffix"][suffix]
			else:
				suffixIdx = numCommonDNSSuffix
			suffixList = [0] * (numCommonDNSSuffix+1)
			suffixList[suffixIdx] = 1
			#3 alexa
			alexaList = [0]*6
			alexa = d['rank']
			if alexa == 100:
				alexaIdx = 0
			elif alexa == 1000:
				alexaIdx = 1
			elif alexa == 10000:
				alexaIdx = 2
			elif alexa == 100000:
				alexaIdx = 3
			elif alexa == 1000000:
				alexaIdx = 4
			else:
				alexaIdx = 5
			alexaList[alexaIdx] = 1
			#4 length of domain name
			lenDN = d['len']
			#5 num of numerical characters
			nNum = d['num']
			#6 num of wildcard or dot
			nNonAlphaNum = d['nonnum']
			#7 num of IPs returned
			nIPs = d['ipCount']
			#integration of all the DNS features
			#concatDNS = ttlList + suffixList + alexaList + [lenDN] + [nNum] + [nNonAlphaNum] + [nIPs]
			concatDNS = ttlList + suffixList + alexaList + [lenDN/float(100)] + [nNum/float(10)] + [nNonAlphaNum/float(10)] + [nIPs/float(10)]
			dnsDict[ip] = concatDNS
	return dnsDict

#for intialization of the number of TLS features
#must be aligned with the calculation below!
def initNumTLSFeature():
	global impact
	global numTLSFeature
	# client Ciphersuite + client Extension + server Ciphersuite + server Extension + \
	# client Public Key Length + number of server certificates + number of SAN names + validity in days + whether self-signed
	numTLSFeature = numCommonTLSClientCS + numCommonTLSClientExt + numCommonTLSServerCS + numCommonTLSServerExt + 1 + 1 + 1 + 1 + 1
	print("numTLSFeature: " + str(numTLSFeature))
	# client Ciphersuite
	tmp = [""] * numCommonTLSClientCS
	for key in tlsCommon["clientCS"].keys():
		index = tlsCommon["clientCS"][key]
		tmp[index] = "clientCS_" + key
	impact += tmp
	# client Extension
	tmp = [""] * numCommonTLSClientExt
	for key in tlsCommon["clientExt"].keys():
		index = tlsCommon["clientExt"][key]
		tmp[index] = "clientExt_" + key
	impact += tmp
	# server Ciphersuite
	tmp = [""] * numCommonTLSServerCS
	for key in tlsCommon["serverCS"].keys():
		index = tlsCommon["serverCS"][key]
		tmp[index] = "serverCS_" + key
	impact += tmp
	# server Extension
	tmp = [""] * numCommonTLSServerExt
	for key in tlsCommon["serverExt"].keys():
		index = tlsCommon["serverExt"][key]
		tmp[index] = "serverExt_" + key
	impact += tmp
	# client Public Key Length
	tmp = ["ClientKeyLen"]
	impact += tmp
	# number of server certificates
	tmp = ["NumServerCertificates"]
	impact += tmp
	# number of SAN names
	tmp = ["NumSubjectAltNames"]
	impact += tmp
	# validity in days
	tmp = ["Validity"]
	impact += tmp
	# whether self-signed
	tmp = ["SelfSigned"]
	impact += tmp


def processTLS(tls):
	global numCommonTLSClientCS
	global numCommonTLSClientExt
	global numCommonTLSServerCS
	global numCommonTLSServerExt
	tlsDict = defaultdict()
	for key in tls.keys():
		if key == "totalTLS":
			continue
		#1 client Ciphersuite 
		clientCSList = [0] * numCommonTLSClientCS
		clientCS = tls[key]['clientCS']
		for cs in clientCS:
			csIdx = tlsCommon["clientCS"][cs]
			clientCSList[csIdx] = 1
		#2 client Extension
		clientExtList = [0] * numCommonTLSClientExt
		clientExt = tls[key]['clientExt']
		for ext in clientExt:
			extIdx = tlsCommon["clientExt"][ext]
			clientExtList[extIdx] = 1
		#3 server Ciphersuite 
		serverCSList = [0] * numCommonTLSServerCS
		serverCS = tls[key]['serverCS']
		for cs in serverCS:
			csIdx = tlsCommon["serverCS"][cs]
			serverCSList[csIdx] = 1
		#4 server Extension
		serverExtList = [0] * numCommonTLSServerExt
		serverExt = tls[key]['serverExt']
		for ext in serverExt:
			extIdx = tlsCommon["serverExt"][ext]
			serverExtList[extIdx] = 1
		#5 client Public Key Length
		cKeyLen = tls[key]['clientKeyLen']
		#6 number of server certificates
		numCert = tls[key]['certCount']
		#7 number of SAN names
		numSAN = max(tls[key]['certSubAltNames']) if tls[key]['certSubAltNames'] != [] else 0 
		#8 validity in days
		numValid = max(tls[key]['certValidDays']) if tls[key]['certValidDays'] != [] else 0 
		#9 whether self-signed
		selfSigned = tls[key]['certSelfSigned']
		# Integration of all TLS features
		#tlsConcat = clientCSList + clientExtList + serverCSList + serverExtList + [cKeyLen] + [numCert] + [numSAN] + [numValid] + [selfSigned]
		tlsConcat = clientCSList + clientExtList + serverCSList + serverExtList + [cKeyLen/float(1000)] + [numCert/float(10)] + [numSAN/float(100)] + [numValid/float(10000)] + [selfSigned]
		tlsDict[key] = tlsConcat
	return tlsDict

#for intialization of the number of HTTP features
#must be aligned with the calculation below!
def initNumHTTPFeature():
	global impact
	global numHTTPFeature
	#presence + content-type + user-agent + server + code
	numHTTPFeature = (numCommonHTTPPresence+1) + (numCommonHTTPContentType+1) + (numCommonHTTPUserAgent+1) + (numCommonHTTPServer+1) + (numCommonHTTPCode+1)
	print("numHTTPFeature: " + str(numHTTPFeature))
	#presence
	tmp = [""] * (numCommonHTTPPresence+1)
	for key in httpCommon["presence"].keys():
		index = httpCommon["presence"][key]
		tmp[index] = "presence_" + key
	tmp[numCommonHTTPPresence] = "presence_Other" 
	impact += tmp
	#content-type
	tmp = [""] * (numCommonHTTPContentType+1)
	for key in httpCommon["content-type"].keys():
		index = httpCommon["content-type"][key]
		tmp[index] = "content-type_" + key
	tmp[numCommonHTTPContentType] = "content-type_Other"
	impact += tmp
	#user-agent
	tmp = [""] * (numCommonHTTPUserAgent+1)
	for key in httpCommon["user-agent"].keys():
		index = httpCommon["user-agent"][key]
		tmp[index] = "user-agent_" + key
	tmp[numCommonHTTPUserAgent] = "user-agent_Other"
	impact += tmp
	#server
	tmp = [""] * (numCommonHTTPServer+1)
	for key in httpCommon["server"].keys():
		index = httpCommon["server"][key]
		tmp[index] = "server_" + key
	tmp[numCommonHTTPServer] = "server_Other" 
	impact += tmp
	#code
	tmp = [""] * (numCommonHTTPCode+1)
	for key in httpCommon["code"].keys():
		index = httpCommon["code"][key]
		tmp[index] = "code_" + key
	tmp[numCommonHTTPCode] = "code_Other"
	impact += tmp

def processHTTP(http):	
	global numCommonHTTPPresence
	global numCommonHTTPContentType
	global numCommonHTTPUserAgent
	global numCommonHTTPServer
	global numCommonHTTPCode
	httpTime = OrderedDict()
	for timeline in http.keys():
		if timeline == "totalHTTP":
			continue
		#presence
		presenceList = [0] * (numCommonHTTPPresence+1)
		for field in http[timeline].keys():
			if field == "totalHTTP":
				continue
			if field in httpCommon["presence"]:
				index = httpCommon["presence"][field]
				presenceList[index] = 1
			else:
				presenceList[numCommonHTTPPresence] = 1
		#content-type
		contentTypeList = [0] * (numCommonHTTPContentType+1)
		if "content-type" in http[timeline]:
			for item in http[timeline]["content-type"].keys():
				if item in httpCommon["content-type"]:
					index = httpCommon["content-type"][item]
					contentTypeList[index] = 1
				else:
					contentTypeList[numCommonHTTPContentType] = 1
		#user-agent
		userAgentList = [0] * (numCommonHTTPUserAgent+1)
		if "user-agent" in http[timeline]:
			for item in http[timeline]["user-agent"].keys():
				if item in httpCommon["user-agent"]:
					index = httpCommon["user-agent"][item]
					userAgentList[index] = 1
				else:
					userAgentList[numCommonHTTPUserAgent] = 1
		#server
		serverList = [0] * (numCommonHTTPServer+1)
		if "server" in http[timeline]:
			for item in http[timeline]["server"].keys():
				if item in httpCommon["server"]:
					index = httpCommon["server"][item]
					serverList[index] = 1
				else:
					serverList[numCommonHTTPServer] = 1
		#code
		codeList = [0] * (numCommonHTTPCode+1)
		if "code" in http[timeline]:
			for item in http[timeline]["code"].keys():
				if item in httpCommon["code"]:
					index = httpCommon["code"][item]
					codeList[index] = 1
				else:
					codeList[numCommonHTTPCode] = 1
		#Integration of all HTTP features
		httpConcat = presenceList + contentTypeList + userAgentList + serverList + codeList
		httpTime[timeline] = httpConcat
	return httpTime

#tls is the iteration standard
def mergeFeatures(tls, tlsDict, dnsDict, httpDict, meta):
	global numDNSFeature
	global numTLSFeature
	global numHTTPFeature
	global numTimesFeature
	global numLengthsFeature
	global numDistFeature
	global timeScale
	global filterOut
	feature = []
	for key in tls.keys():
		if key == "totalTLS":
			continue
		dAddr = (key.split('@'))[-1]
		if filterOut != [] and dAddr in filterOut:
			#print "skip " + dAddr
			continue
		if timeScale == 0:
			timeline = str(0)
		else:
			timeline = str((tls[key]["ts_start"]) / (timeScale*60) * (timeScale*60))
		tmp = []
		#tls
		if enableTLS == True:
			tmp += tlsDict[key]
		#dns
		if dnsDir != "":
			if dAddr in dnsDict:
				tmp += dnsDict[dAddr]
			else:
				#tmp += [0]*numDNSFeature
				continue
		#http
		if httpDir != "":
			if timeline in httpDict:
				tmp += httpDict[timeline]
			else:
				continue
		#meta
		if timesDir != "":
			tmp += meta[key]["flowTimes"]
		if lengthsDir != "":
			tmp += meta[key]["flowLengths"]
		if distDir != "":
			tmp += meta[key]["flowByteDist"]
		assert(len(tmp) == (numDNSFeature + numTLSFeature + numHTTPFeature + numTimesFeature + numLengthsFeature + numDistFeature))
		feature.append(tmp)
	return feature

def prepFeature(tls, dns, http, meta):
	#tls
	if enableTLS == True:
		tlsDict = processTLS(tls)
	else:
		tlsDict = defaultdict()
	#dns
	if dnsDir != "":
		dnsDict = processDNS(dns)
	else:
		dnsDict = defaultdict()
	#http
	if httpDir != "":
		httpDict = processHTTP(http)
	else:
		httpDict = OrderedDict()
	feature = mergeFeatures(tls, tlsDict, dnsDict, httpDict, meta)
	return feature

def prepData(select, dataType):
	data = []
	for dataset in select:
		#print('Prepare data for %s' %(dataset)) #verbose
		#tls
		tlsFile = tlsDir + dataset + "_TLS.json"
		with open(tlsFile, 'r') as fpTLS:
			tls = json.load(fpTLS)
			#This is possible now, due to the segfault of Joy
			if tls == {}:
				continue
		#dns
		if dnsDir != "":
			dnsFile = dnsDir + dataset + "_DNS.json"
			with open(dnsFile, 'r') as fpDNS:
				dns = json.load(fpDNS)
		else:
			dns = {}
		#http
		if httpDir != "":
			httpFile = httpDir + dataset + "_"+str(timeScale) + "_HTTP.json"
			with open(httpFile, 'r') as fpHTTP:
				http = json.load(fpHTTP)
				http = OrderedDict(sorted(http.items(), key=lambda t: t[0]))
		else:
			http = OrderedDict()
		#meta
		if metaDir != "":
			metaFile = metaDir + dataset + "_META.json"
			with open(metaFile, 'r') as fpMETA:
				meta = json.load(fpMETA)
		else:
			meta = {}
		#Get all the features from TLS and DNS
		data += prepFeature(tls, dns, http, meta)
	# Generate label based upon benign or malware
	label = []
	if dataType == 0:
		label = [0] * len(data)
	elif dataType == 1:
		label = [1] * len(data)
	return (data, label)

def pullData(select):
	selectMap = json.load(open(select, 'r'))
	pos = prepData(selectMap["Benign"], 0)
	print('positive examples: %s' %(str(len(pos[0]))))
	neg = prepData(selectMap["Malware"], 1)
	print('negative examples: %s' %(str(len(neg[0]))))
	data = pos[0] + neg[0]
	label = pos[1] + neg[1]
	return (data, label)


def main():
	startTime = time.time()
	# Read in common files
	global dnsCommon
	global tlsCommon
	global httpCommon 
	
	## Get the number of the most Common Features, since it is copied from these other files
	#DNS
	global dnsDir
	global dnsCommonPath
	global numDNSFeature
	global numCommonDNSTTL
	global numCommonDNSSuffix
	
	#TLS
	global tlsDir
	global tlsCommonPath
	global enableTLS
	global numTLSFeature
	global numCommonTLSClientCS
	global numCommonTLSClientExt
	global numCommonTLSServerCS
	global numCommonTLSServerExt
	
	#HTTP
	global httpDir
	global httpCommonPath
	global numHTTPFeature
	global numCommonHTTPPresence
	global numCommonHTTPContentType
	global numCommonHTTPUserAgent
	global numCommonHTTPServer
	global numCommonHTTPCode
	
	#META
	global metaDir
	global timesDir
	global lengthsDir
	global distDir
	global numTimesFeature
	global numLengthsFeature
	global numDistFeature
	#Impact
	global impact
	#timeScale
	global timeScale
	#filterOut
	global filterOut
	
	## setup parser
	parser = argparse.ArgumentParser(description="Classify the Flows Based Upon DNS and TLS Features", add_help=True)
	parser.add_argument('--workDir', action="store", help="The directory where we store the feature data")
	parser.add_argument('--select', action="store", help="The VALID selection JSON file with \
		                                            both key Malware and key Benign, \
		                                            values are list of datasets")
	parser.add_argument('--input', action="store", help="Test Input Parameters File")
	parser.add_argument('--output', action="store", help="Training Output Parameters File")
	parser.add_argument('--test', action="store_true", default=False, help="Whether Test")
	parser.add_argument('--analyze', action="store_true", default=False, help="Whether only analyze params file without real test")
	parser.add_argument('--classify', action="store_true", default=False, help="Whether Classify")
	parser.add_argument('--model', action="store", help="The machine learning model including LR, SVM and ANN")
	parser.add_argument('--dns', action="store_true", default=False, help="Whether enable DNS feature")
	parser.add_argument('--http', action="store_true", default=False, help="Whether enable HTTP feature")
	parser.add_argument('--tls', action="store_true", default=False, help="Whether enable TLS feature")
	parser.add_argument('--times', action="store_true", default=False, help="Whether enable inter-packet times feature")
	parser.add_argument('--lengths', action="store_true", default=False, help="Whether enable inter-packet lengths feature")
	parser.add_argument('--dist', action="store_true", default=False, help="Whether enable byte distribution feature")
	parser.add_argument('--timeScale', action="store", help="The granularity of time window of HTTP flow in minutes")
	parser.add_argument('--filterOut', action="store", help="The filter-out JSON file")
	args = parser.parse_args()
	#command example
	#python classify.py --workDir=./ --select=./selection/testc.json --classify --output=params.txt  --tls --dns --http
	#python classify.py --workDir=./ --select=./selection/testc.json --test --input=params.txt
	#check the time scale for http flows
	if args.timeScale == None:
		timeScale = 0
	else:
		timeScale = args.timeScale
	#check if the filter exists
	if args.filterOut != None:
		if not os.path.isfile(args.filterOut):
			print("No valid fiter-out JSON file")
			return
		else:
			filterOut = json.load(open(args.filterOut, 'r'))
			filterOut = set(filterOut)
	#check the work directory
	if args.workDir == None or not os.path.isdir(args.workDir):
		print('No valid work directory')
		return
	else:
		workDir = args.workDir
		if not workDir.endswith('/'):
			workDir += '/'
	#Setup common feature paths
	tlsCommonPath = workDir + "TLS_COMMON/tlsCommon.json"
	if not os.path.isfile(tlsCommonPath):
		print("No valid tlsCommon.json path")
		return
	with open(tlsCommonPath, 'r') as fpTLS:
		tlsCommon = json.load(fpTLS)
	
	dnsCommonPath = workDir + "DNS_COMMON/dnsCommon.json"
	if not os.path.isfile(dnsCommonPath):
		print("No valid dnsCommon.json path")
		return
	with open(dnsCommonPath, 'r') as fpDNS:
		dnsCommon = json.load(fpDNS)
	
	httpCommonPath = workDir + "HTTP_COMMON/httpCommon.json"
	if not os.path.isfile(httpCommonPath):
		print("No valid httpCommon.json path")
		return
	with open(httpCommonPath, 'r') as fpHTTP:
		httpCommon = json.load(fpHTTP)
	#Setup corresponding features
	numCommonDNSTTL = len(list(dnsCommon["ttls"].keys()))
	numCommonDNSSuffix = len(list(dnsCommon["suffix"].keys()))
	numCommonTLSClientCS = len(list(tlsCommon["clientCS"].keys()))
	numCommonTLSClientExt = len(list(tlsCommon["clientExt"].keys()))
	numCommonTLSServerCS = len(list(tlsCommon["serverCS"].keys()))
	numCommonTLSServerExt = len(list(tlsCommon["serverExt"].keys()))
	numCommonHTTPPresence = len(list(httpCommon["presence"].keys()))
	numCommonHTTPContentType = len(list(httpCommon["content-type"].keys()))
	numCommonHTTPUserAgent = len(list(httpCommon["user-agent"].keys()))
	numCommonHTTPServer = len(list(httpCommon["server"].keys()))
	numCommonHTTPCode = len(list(httpCommon["code"].keys()))

	#We use the TLS_JSON as the iteration standard, hence TLS_JSON directory must exist
	tlsDir = workDir + "TLS_JSON/" 
	if not os.path.isdir(tlsDir):
		print("No valid TLS_JSON directory")
		return
	#Check for classify and test
	if args.classify == False and args.test == False:
		print('At least test or classify')
		return
	if args.classify == True and args.test == True:
		print('Classify and Test cannot work at the same time')
		return
	if args.test == True and (args.input == None or os.path.isdir(args.input)):
		print('Test Needs Input Parameters File')
		return
	if args.classify == True and (args.output == None or os.path.isdir(args.output)):
		print('Classify Needs Output Parameters File')
		return

	#Classify Only
	if args.classify == True:
		#Check if select is enabled
		if args.select == None or not os.path.isfile(args.select):
			print('No valid selection JSON file')
			return

		#check if at least one of the features is enabled
		if args.dns == False and args.http == False and args.tls == False and args.times == False and args.lengths == False and args.dist == False:
			print("At least one feature is required!")
			return

		#tls feature
		if args.tls == True:
			enableTLS = True
			initNumTLSFeature()

		#dns feature  
		if args.dns == True:
			if not os.path.isdir(workDir + "DNS_JSON/"):
				print("No valid DNS_JSON directory")
				return
			else:
				dnsDir = workDir + "DNS_JSON/"
				initNumDNSFeature()
		
		#http feature
		if args.http == True: 
			if not os.path.isdir(workDir + "HTTP_JSON/"):
				print("No valid HTTP_JSON directory")
				return
			else:
				httpDir = workDir + "HTTP_JSON/"
				initNumHTTPFeature()

		#times, lengths and byte-distribution
		if args.times == True or args.lengths == True or args.dist == True:
			metaDir = workDir + "META_JSON/"
			if not os.path.isdir(metaDir):
				print("No valid META_JSON directory")
				return
			else:
				if args.times == True:
					timesDir = metaDir
					initNumTimesFeature()
				if args.lengths == True:
					lengthsDir = metaDir
					initNumLengthsFeature()
				if args.dist == True:
					distDir = metaDir
					initNumDistFeature()

		#check the model
		if args.model == None:
			mlModel = LR(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
		else:
			if args.model == "LR":
				mlModel = LR(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
			elif args.model == "SVM":
				mlModel = SVM(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
			elif args.model == "ANN":
				mlModel = ANN(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
				'''
			elif args.model == "ANNGPU":
				mlModel = ANNGPU(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
				'''
			elif args.model == "DF":
				mlModel = DF(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)	
			elif args.model == "vinoANN":
				mlModel = vinoANN(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
			else:
				print(args.model + " is not supported!")
				return

		#Get the flow feature data
		processStart = time.time()
		d = pullData(args.select)
		processEnd = time.time()
		print("Data prep elapsed in %.3f seconds" %(processEnd - processStart))
		# train the model
		acc = mlModel.train(d[0], d[1], args.output, workDir)
		if args.model != 'vinoANN':
			print("Accuracy: %.3f%% train, %.3f%% test" %(acc[0], acc[1]))
	#Test only
	else:
		processStart = time.time()
		inputParam = "%s%s" % (workDir, args.input)
		paramMap = json.load(open(inputParam, 'r'))
		numTLSFeatureT = paramMap["feature"]["tls"]
		numDNSFeatureT = paramMap["feature"]["dns"]
		numHTTPFeatureT = paramMap["feature"]["http"]
		numTimesFeatureT = paramMap["feature"]["times"]
		numLengthsFeatureT = paramMap["feature"]["lengths"]
		numDistFeatureT = paramMap["feature"]["dist"]
		#tls feature
		if numTLSFeatureT != 0:
			enableTLS = True
			initNumTLSFeature()
			assert(numTLSFeatureT == numTLSFeature)

		#dns feature  
		if numDNSFeatureT != 0:
			if not os.path.isdir(workDir + "DNS_JSON/"):
				print("No valid DNS_JSON directory")
				return
			else:
				dnsDir = workDir + "DNS_JSON/"
				initNumDNSFeature()
				assert(numDNSFeatureT == numDNSFeature)
		
		#http feature
		if numHTTPFeatureT != 0: 
			if not os.path.isdir(workDir + "HTTP_JSON/"):
				print("No valid HTTP_JSON directory")
				return
			else:
				httpDir = workDir + "HTTP_JSON/"
				initNumHTTPFeature()
				assert(numHTTPFeatureT == numHTTPFeature)

		#times, lengths and byte-distribution
		if numTimesFeatureT != 0 or numLengthsFeatureT != 0 or numDistFeatureT != 0:
			metaDir = workDir + "META_JSON/"
			if not os.path.isdir(metaDir):
				print("No valid META_JSON directory")
				return
			else:
				if numTimesFeatureT != 0:
					timesDir = metaDir
					initNumTimesFeature()
					assert(numTimesFeatureT == numTimesFeature)
				if numLengthsFeatureT != 0:
					lengthsDir = metaDir
					initNumLengthsFeature()
					assert(numLengthsFeatureT == numLengthsFeature)
				if numDistFeatureT != 0:
					distDir = metaDir
					initNumDistFeature()
					assert(numDistFeatureT == numDistFeature)

		#check the model
		if args.model == None:
			mlModel = LR(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
		else:
			if args.model == "LR":
				mlModel = LR(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
			elif args.model == "SVM":
				mlModel = SVM(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
			elif args.model == "ANN":
				mlModel = ANN(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
				'''
			elif args.model == "ANNGPU":
				mlModel = ANNGPU(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
				'''
			elif args.model == "DF":
				mlModel = DF(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)	
			elif args.model == "vinoANN":
				mlModel = vinoANN(numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature)
			else:
				print(args.model + " is not supported!")
				return

		#if args.analyze == False:
		#Get the flow feature data
		d = pullData(args.select)
		processEnd = time.time()
		print("Data prep elapsed in %.3f seconds" %(processEnd - startTime))
		# test the model
		acc = mlModel.test(np.array(d[0]), d[1], workDir)
		if args.model != "vinoANN":
			print("Testing accuracy: %.3f%%" %(acc))
		'''
		else:
			assert(len(impact) == len(paramMap["coef_"]))
			threshHold = 0
			pos = []
			neg = []
			for i in range(0, len(impact)):
				val = paramMap["coef_"][i]
				if abs(val) > threshHold:
					if val > 0:
						pos.append((impact[i], val))
					else:
						neg.append((impact[i], val))
			pos.sort(key=lambda x: x[1], reverse=True)
			neg.sort(key=lambda x: x[1], reverse=False)
			print("pos impact: ")
			print pos
			print("neg impact: ")
			print neg
		'''
	endTime = time.time()
	print("Classify elapsed in %.3f seconds" %(endTime - startTime))


if __name__ == "__main__":
	main()
