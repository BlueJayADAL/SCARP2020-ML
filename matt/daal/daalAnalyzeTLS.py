import ujson as json
import sys
import gzip
from collections import defaultdict
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# for extension
extMap = {
    "server_name": 0, "max_fragment_length" : 1, "client_certificate_url" : 2, "trusted_ca_keys": 3,
    "truncated_hmac": 4, "status_request" : 5, "user_mapping" : 6, "client_authz" : 7,
	"server_authz": 8, "cert_type" : 9, "supported_groups" : 10, "ec_point_formats" : 11,
    "srp" : 12, "signature_algorithms" : 13, "use_srtp" : 14, "heartbeat" : 15,
    "application_layer_protocol_negotiation" : 16, "status_request_v2" : 17, "signed_certificate_timestamp" : 18, "client_certificate_type" : 19,
    "server_certificate_type" : 20, "padding" : 21, "encrypt_then_mac" : 22, "extended_master_secret" : 23,
    "token_binding" : 24, "cached_info" : 25, "session_ticket" : 35, "renegotiation_info" : 65281
}
def obtainExtList(cexts):
	exts = []
	for item in cexts:
		if "kind" in item:
			exts.append("%0.4x" %(int(item["kind"])))
		else:
			exts.append("%0.4x" %(extMap[list((item.keys()))[0]]))
	return exts

#for valid days
#format ASN1_TIME: Aug 8 12:49:31 2017 GMT
daysPassed = {}
daysPassed["Jan"] = 0
daysPassed["Feb"] = 31
daysPassed["Mar"] = 59
daysPassed["Apr"] = 90
daysPassed["May"] = 120
daysPassed["Jun"] = 151
daysPassed["Jul"] = 181
daysPassed["Aug"] = 212
daysPassed["Sep"] = 243
daysPassed["Oct"] = 273
daysPassed["Nov"] = 304
daysPassed["Dec"] = 334
def obtainDay(tstamp):
	t = tstamp.split()
	month = t[0]
	day = int(t[1])
	year = int(t[3])
	return daysPassed[month]+day+(year-2000)*365

def obtainValidList(scerts):
	valid = []
	for item in scerts:
		if ("validity_not_before" in item) and ("validity_not_after" in item):
			before = item["validity_not_before"]
			after = item["validity_not_after"]
			t1 = obtainDay(before)
			t2 = obtainDay(after)
			valid.append(t2-t1)
	return valid

def obtainSelfSigned(scerts):
	if len(scerts) > 1 or len(scerts) == 0:
		return 0
	elif ("issuer" in scerts[0]) and ("subject" in scerts[0]):
		return scerts[0]["issuer"] == scerts[0]["subject"]
	else:
		return 0

def obtainSANs(scerts):
	try:
		sans = []
		for cert in scerts:
			exts = cert["extensions"]
			for ext in exts:
				key = list((ext.keys()))[0]
				if key.find("Subject Alternative Name") != -1:
					sans.append(len(ext[key].split(',')))	
		return sans
	except:
		return []

def ProcessTLS(inPathName, fileName, tls):
	json_file = "%s%s" % (inPathName, fileName)
	#print("processing TLS for %s" %(json_file)) #verbose
	#read each line and convert it into dict
	lineno = 0
	total = 0
	with gzip.open(json_file,'r') as fp:
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
			resp = tmp["tls"]
			try:
				tls[serverAddr]['count'] += 1
			#if serverAddr not in tls:
			except KeyError:
				tls[serverAddr] = defaultdict()
				tls[serverAddr]['count'] = 1
				tls[serverAddr]['ts_start'] = tmp["time_start"]
				tls[serverAddr]['ts_end'] = tmp["time_end"]
				#1. client-offered ciphersuites
				if "cs" in resp:
					tls[serverAddr]['clientCS'] = resp["cs"] 
				else:
					tls[serverAddr]['clientCS'] = []
				#2. client-advertised extensions
				if "c_extensions" in resp:
					tls[serverAddr]['clientExt'] = obtainExtList(resp["c_extensions"])
				else:
					tls[serverAddr]['clientExt'] = []
				#3. client-public key length
				if "c_key_length" in resp:	
					tls[serverAddr]['clientKeyLen'] = resp["c_key_length"]-16
				else:
					tls[serverAddr]['clientKeyLen'] = 0

				#4. server-selected ciphersuite
				if "scs" in resp:
					temp = []
					temp.append(resp["scs"])
					tls[serverAddr]['serverCS'] = temp #list of strings
				else:
					tls[serverAddr]['serverCS'] = []
				#5. server-supported extensions
				if "s_extensions" in resp:
					tls[serverAddr]['serverExt'] = obtainExtList(resp["s_extensions"])
				else:
					tls[serverAddr]['serverExt'] = []
				if "s_cert" in resp:
					cert = resp["s_cert"]
					#6. number of certificates
					tls[serverAddr]['certCount'] = len(cert)
					#7. number of validity in days
					tls[serverAddr]['certValidDays'] = obtainValidList(cert) #list of validity
					#8. whether self-signed
					tls[serverAddr]['certSelfSigned'] = obtainSelfSigned(cert)
					#9. number of subject alternative names
					tls[serverAddr]['certSubAltNames'] = obtainSANs(cert)
				else:
					tls[serverAddr]['certCount'] = 0
					tls[serverAddr]['certValidDays'] = []
					tls[serverAddr]['certSelfSigned'] = 0
					tls[serverAddr]['certSubAltNames'] = []

	try:
		tls["totalTLS"] += total	
	#if "totalTLS" not in tls:
	except KeyError:
		tls["totalTLS"] = total
		

def saveToJson(outPathName, fileName, tls):
	fname = "%s%s_TLS.json" % (outPathName, (fileName.split('.'))[0])
	#print("save JSON to " + fname) #verbose
	with open(fname, 'w') as fp:
		json.dump(tls, fp) 

def plotTLS(tls, inPathName, fileName, outPathName):
	outFolder = outPathName + (fileName.split('.'))[0] + "/"
	if not os.path.exists(outFolder):
		os.mkdir(outFolder)
	
	#1. client-offered ciphersuites
	csDict = {}
	csList = []
	total = 0
	for tup in list(tls.keys()):
		total += 1
		for cs in tls[tup]['clientCS']:
			if cs in csDict:
				csDict[cs] += 1
			else:
				csDict[cs] = 1
	for cs in list(csDict.keys()):
		csList.append( (cs, csDict[cs]/float(total)*100) )
	csList.sort(key=lambda x: x[1], reverse=True)
	if len(csList) > 15:
		csList = csList[0:14]
	plt.clf()
	plt.bar(range(0, len(csList)), [x[1] for x in csList], align='center', alpha=0.5)
	plt.xticks(range(0, len(csList)), [x[0] for x in csList], rotation=45)
	plt.ylabel('Percentage of Flows')
	plt.title('Offered Ciphersuites')
	plt.savefig(outFolder+"clientCS.pdf")

	#2. client-advertised extensions
	extDict = {}
	extList = []
	total = 0
	for tup in list(tls.keys()):
		total += 1
		for ext in tls[tup]['clientExt']:
			if ext in extDict:
				extDict[ext] += 1
			else:
				extDict[ext] = 1
	for ext in list(extDict.keys()):
		extList.append( (ext, extDict[ext]/float(total)*100) )
	extList.sort(key=lambda x: x[1], reverse=True)
	if len(extList) > 15:
		extList = extList[0:14]
	plt.clf()
	plt.bar(range(0, len(extList)), [x[1] for x in extList], align='center', alpha=0.5)
	plt.xticks(range(0, len(extList)), [x[0] for x in extList], rotation=45)
	plt.ylabel('Percentage of Flows')
	plt.title('Advertised TLS Extensions')
	plt.savefig(outFolder+"clientExt.pdf")

	#3. client-public key length
	itemDict = {}
	itemList = []
	total = 0
	for tup in list(tls.keys()):
		total += 1
		item = tls[tup]['clientKeyLen']
		if item in itemDict:
			itemDict[item] += 1
		else:
			itemDict[item] = 1

	for item in list(itemDict.keys()):
		itemList.append( (item, itemDict[item]/float(total)*100) )
	itemList.sort(key=lambda x: x[1], reverse=True)
	if len(itemList) > 15:
		itemList = itemList[0:14]
	plt.clf()
	plt.bar(range(0, len(itemList)), [x[1] for x in itemList], align='center', alpha=0.5)
	plt.xticks(range(0, len(itemList)), [x[0] for x in itemList], rotation=45)
	plt.ylabel('Percentage of Flows')
	plt.title('Public Key Length')
	plt.savefig(outFolder+"clientKeyLen.pdf")


	#4. server-selected ciphersuite
	csDict = {}
	csList = []
	total = 0
	for tup in list(tls.keys()):
		total += 1
		for cs in tls[tup]['serverCS']:
			if cs in csDict:
				csDict[cs] += 1
			else:
				csDict[cs] = 1
	for cs in list(csDict.keys()):
		csList.append( (cs, csDict[cs]/float(total)*100) )
	csList.sort(key=lambda x: x[1], reverse=True)
	if len(csList) > 15:
		csList = csList[0:14]
	plt.clf()
	plt.bar(range(0, len(csList)), [x[1] for x in csList], align='center', alpha=0.5)
	plt.xticks(range(0, len(csList)), [x[0] for x in csList], rotation=45)
	plt.ylabel('Percentage of Flows')
	plt.title('Supported Ciphersuites')
	plt.savefig(outFolder+"serverCS.pdf")

	#5. server-supported extensions
	extDict = {}
	extList = []
	total = 0
	for tup in list(tls.keys()):
		total += 1
		for ext in tls[tup]['serverExt']:
			if ext in extDict:
				extDict[ext] += 1
			else:
				extDict[ext] = 1
	for ext in list(extDict.keys()):
		extList.append( (ext, extDict[ext]/float(total)*100) )
	extList.sort(key=lambda x: x[1], reverse=True)
	if len(extList) > 15:
		extList = extList[0:14]
	plt.clf()
	plt.bar(range(0, len(extList)), [x[1] for x in extList], align='center', alpha=0.5)
	plt.xticks(range(0, len(extList)), [x[0] for x in extList], rotation=45)
	plt.ylabel('Percentage of Flows')
	plt.title('Supported TLS Extensions')
	plt.savefig(outFolder+"serverExt.pdf")
	 
	#6. number of certificates
	certDict = {}
	certList = []
	total = 0
	for tup in list(tls.keys()):
		total += 1
		item = tls[tup]['certCount']
		if item in certDict:
			certDict[item] += 1
		else:
			certDict[item] = 1
	if 1 in certDict:
		toalOneChain = certDict[1] #for self signed
	else:
		toalOneChain = total #for self signed
	for item in list(certDict.keys()):
		certList.append( (item, certDict[item]/float(total)*100) )
	certList.sort(key=lambda x: x[1], reverse=True)
	if len(certList) > 15:
		certList = certList[0:14]
	plt.clf()
	plt.bar(range(0, len(certList)), [x[1] for x in certList], align='center', alpha=0.5)
	plt.xticks(range(0, len(certList)), [x[0] for x in certList], rotation=45)
	plt.ylabel('Percentage of Flows')
	plt.title('Number of Certificates')
	plt.savefig(outFolder+"certCount.pdf")

	#7. number of validity in days
	certDict = {}
	certList = []
	total = 0
	for tup in list(tls.keys()):
		total += 1
		for item in tls[tup]['certValidDays']:
			if item in certDict:
				certDict[item] += 1
			else:
				certDict[item] = 1
	for item in list(certDict.keys()):
		certList.append( (item, certDict[item]/float(total)*100) )
	certList.sort(key=lambda x: x[1], reverse=True)
	if len(certList) > 15:
		certList = certList[0:14]
	plt.clf()
	plt.bar(range(0, len(certList)), [x[1] for x in certList], align='center', alpha=0.5)
	plt.xticks(range(0, len(certList)), [x[0] for x in certList], rotation=45)
	plt.ylabel('Percentage of Flows')
	plt.title('Validity (in Days)')
	plt.savefig(outFolder+"certValidDays.pdf")

	#8. whether self-signed
	certDict = {}
	certList = []
	total = 0
	for tup in list(tls.keys()):
		total += 1
		item = tls[tup]['certSelfSigned']
		if item in certDict:
			certDict[item] += 1
		else:
			certDict[item] = 1

	for item in list(certDict.keys()):
		if item == 0:
			certList.append( (item, certDict[item]/float(total)*100) )
		else:
			certList.append( (item, certDict[item]/float(toalOneChain)*100) )
	certList.sort(key=lambda x: x[1], reverse=True)
	if len(certList) > 15:
		certList = certList[0:14]
	plt.clf()
	plt.bar(range(0, len(certList)), [x[1] for x in certList], align='center', alpha=0.5)
	plt.xticks(range(0, len(certList)), [x[0] for x in certList], rotation=45)
	plt.ylabel('Percentage of Flows in 1-Chain')
	plt.title('Self Signed')
	plt.savefig(outFolder+"certSelfSigned.pdf")

	#9. number of subject alternative names
	certDict = {}
	certList = []
	total = 0
	for tup in list(tls.keys()):
		total += 1
		for item in tls[tup]['certSubAltNames']:
			if item in certDict:
				certDict[item] += 1
			else:
				certDict[item] = 1
	for item in list(certDict.keys()):
		certList.append( (item, certDict[item]/float(total)*100) )
	certList.sort(key=lambda x: x[1], reverse=True)
	if len(certList) > 15:
		certList = certList[0:14]
	plt.clf()
	plt.bar(range(0, len(certList)), [x[1] for x in certList], align='center', alpha=0.5)
	plt.xticks(range(0, len(certList)), [x[0] for x in certList], rotation=45)
	plt.ylabel('Percentage of Flows')
	plt.title('Number of SubjectAltNames')
	plt.savefig(outFolder+"certSubAltNames.pdf")


def main():
	parser = argparse.ArgumentParser(description="Probability Distribution of TLS Features in Dataset", add_help=True)
	parser.add_argument('-i', '--input', action="store", help="The input folder containing files generated by Joy")
	parser.add_argument('-j', '--json', action="store_true", default=False, help="Generate JSON output file")
	parser.add_argument('-f', '--figure', action="store_true", default=False, help="Generate Probability Distribution Figures")
	parser.add_argument('-a', '--allFile', action="store_true", default=False, help="Indicate whether treat all the file as together or separate")
	args = parser.parse_args()

	#setup input folder and output folders
	if args.input == None or not os.path.isdir(args.input):
		print("No valid input folder!")
		return
	else:
		joyFolder = args.input
		if not joyFolder.endswith('/'):
			joyFolder += '/'
	parentFolder = os.path.abspath(os.path.join(joyFolder, os.pardir))
	if not parentFolder.endswith('/'):
		parentFolder += '/'
	TLS_JSON_Folder = "%sTLS_JSON/" % parentFolder
	TLS_Figure_Folder = "%sTLS_Figure" % parentFolder 
	if not os.path.exists(TLS_JSON_Folder):
		os.mkdir(TLS_JSON_Folder)
	if args.figure:
		if not os.path.exists(TLS_Figure_Folder):
			os.mkdir(TLS_Figure_Folder)

	#check if output JSON
	if args.json:
		if args.allFile == True: 
			tls = defaultdict()
			allFileName = ""
		files = os.listdir(joyFolder)
		for item in files:
			if args.allFile == True: 
				allFileName += (item.split('.'))[0] + "-"
			try:
				if args.allFile == False: 
					tls = defaultdict()
				ProcessTLS(joyFolder, item, tls)
				if args.allFile == False: 
					saveToJson(TLS_JSON_Folder, item, tls)
				if args.figure:
					if args.allFile == False: 
						plotTLS(tls, joyFolder, item, TLS_Figure_Folder)
			except:
				continue
		if args.allFile == True:
			allFileName +=  ".json"
			saveToJson(TLS_JSON_Folder, allFileName, tls)
			if args.figure:
				plotTLS(tls, joyFolder, allFileName, TLS_Figure_Folder)
	
	#check if output figures	
	elif args.figure:
		if args.allFile == True: 
			allFileName = ""
			files = os.listdir(joyFolder)
			for item in files:
				allFileName += (item.split('.'))[0] + "-"
			allFileName +=  ".json"
			fName = TLS_JSON_Folder + (allFileName.split('.'))[0] + "_TLS.json"
			if os.path.exists(fName):
				try:
					with open(fName,'r') as fp:
						tls = json.load(fp)
						plotTLS(tls, joyFolder, allFileName, TLS_Figure_Folder)
				except:
					pass
			else:
				tls = defaultdict()
				for item in files:
					try:
						ProcessTLS(joyFolder, item, tls)
					except:
						continue
				try:
					plotTLS(tls, joyFolder, allFileName, TLS_Figure_Folder)
				except:
					pass
		else:
			files = os.listdir(joyFolder)
			for item in files:
				try:
					fName = TLS_JSON_Folder + (item.split('.'))[0] + "_TLS.json"
					if os.path.exists(fName):
						with open(fName,'r') as fp:
							tls = json.load(fp)
							plotTLS(tls, joyFolder, item, TLS_Figure_Folder)	
					else:
						tls = defaultdict()
						ProcessTLS(joyFolder, item, tls)
						plotTLS(tls, joyFolder, item, TLS_Figure_Folder)
				except:
					continue


if __name__ == "__main__":
	main()

