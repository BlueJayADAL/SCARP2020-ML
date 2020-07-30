import ujson as json
import sys
import gzip
#import seolib as seo
from collections import defaultdict
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import urllib.request, re
import time
# a function to return a domain's rank
# alexaMap dict for caching
alexaMap = {}
def obtainAlexa(url):
	global alexaMap
	if url in alexaMap:
		return alexaMap[url]
	try:
		time.sleep(0.1)
		inurl = "http://data.alexa.com/data?cli=10&url=%s" % (url)
		xml = urllib.request.urlopen(inurl).read().decode('utf-8')
		alexa_rank = int(re.search(r'<POPULARITY[^>]*TEXT="(\d+)"', xml).groups()[0])
	except:
		alexa_rank = None

	if alexa_rank == None:
		alexaMap[url] = alexa_rank
		return alexa_rank
	elif alexa_rank > 1000000:
		alexa_rank = 10000000
	elif alexa_rank > 100000:
		alexa_rank = 1000000
	elif alexa_rank > 10000:
		alexa_rank = 100000
	elif alexa_rank > 1000:
		alexa_rank = 10000
	elif alexa_rank > 100:
		alexa_rank = 1000
	elif alexa_rank > 0:
		alexa_rank = 100

	alexaMap[url] = alexa_rank
	return alexa_rank

# a function to collect the number of numerical character or non-alphanum
def obtainNumAndNonnum(url):
	num = 0
	nonnum = 0
	for c in url:
		if c.isdigit():
			num = num+1
		elif c == '.' or c == '*':
			nonnum = nonnum + 1
	return num, nonnum

# a function to collect the number of IPs and TTL
def obtainIPAndTTL(rr):
	ips = []
	ttls = []
	for item in rr:
		if "a" in item:
			ips.append(item["a"])
			try:
				t = int(item["ttl"])
			except:
				t = 0
			ttls.append(t)
	return (ips, ttls)

def ProcessDNS(inPathName, fileName, dns):
	json_file = "%s%s" % (inPathName, fileName)
	#print("processing DNS for %s" %(json_file)) #verbose
	#read each line and convert it into dict
	total = 0
	with gzip.open(json_file,'r') as fp:  
		for line in fp:
			try:
				tmp = json.loads(line) 
			except:
				continue
			if ('version' in tmp) or ("dns" not in tmp):
				continue
			total += 1
			resp = tmp["dns"][0]
			try:
				rname = resp["rn"]
				rrecord = resp["rr"]
			except:
				continue
			try: 
				dns[rname]['count'] += 1			
			except KeyError:
				dns[rname] = defaultdict()
				dns[rname]['count'] = 1
				#1. length of the query name
				dns[rname]['len'] = len(rname)
				#2. suffixes
				dns[rname]['suffix'] = rname.split('.')[-1]
				first, second = obtainNumAndNonnum(rname)
				#3. # of numerical character
				dns[rname]['num'] = first
				#4. # of wildcards or periods
				dns[rname]['nonnum'] = second
				first, second = obtainIPAndTTL(rrecord)
				dns[rname]['ips'] = first
				#5. # of IP addresses
				dns[rname]['ipCount'] = len(first)
				#6. ttl
				dns[rname]['ttls'] = second
				#7. Alexa rank
				dns[rname]['rank'] = obtainAlexa(rname)

	try:
		dns["totalDNS"] += total
	#if "totalDNS" not in dns: 
	except KeyError:
		dns["totalDNS"] = total
	#else:
		

def saveToJson(outPathName, fileName, dns):
	fname = "%s%s_DNS.json" % (outPathName, (fileName.split('.'))[0])
	#print("save JSON to %s" % (fname)) #verbose
	with open(fname, 'w') as fp:
		json.dump(dns, fp)	

def plotDNSFeature(dns, feature, ylabel, title, outFolder):
	itemDict = {}
	itemList = []
	total = 0
	for tup in dns.keys():
		count = dns[tup]['count']
		total += count
		item = dns[tup][feature]
		if item in itemDict:
			itemDict[item] += count
		else:
			itemDict[item] = count

	for item in itemDict.keys():
		itemList.append( (item, itemDict[item]/float(total)*100) )
	itemList.sort(key=lambda x: x[1], reverse=True)
	if len(itemList) > 15:
		itemList = itemList[0:14]
	plt.clf()
	plt.bar(range(0, len(itemList)), [x[1] for x in itemList], align='center', alpha=0.5)
	plt.xticks(range(0, len(itemList)), [x[0] for x in itemList], rotation=30)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(outFolder + feature + ".pdf")

def plotDNS(dns, inPathName, fileName, outPathName):
	outFolder = outPathName + (fileName.split('.'))[0] + "/"
	if not os.path.exists(outFolder):
		os.mkdir(outFolder)
	#0 top 15 domain names
	itemList = []
	for tup in dns.keys():
		count = dns[tup]['count']
		itemList.append( (tup, count) )
	itemList.sort(key=lambda x: x[1], reverse=True)
	if len(itemList) > 15:
		itemList = itemList[0:14]
	plt.clf()
	plt.tight_layout()
	plt.barh(range(0, len(itemList)), [x[1] for x in itemList], align='center', alpha=0.6)
	plt.yticks(range(0, len(itemList)), [x[0] for x in itemList], rotation=60, fontsize = 5)
	plt.title("Top Domains")
	plt.savefig(outFolder + "count.pdf")
	#1. length of the query name
	plotDNSFeature(dns, 'len', 'Percentage of Flows', 'Num. of Chars in the Domain', outFolder)
	#2. suffixes
	plotDNSFeature(dns, 'suffix', 'Percentage of Flows', 'Suffix of Domain Names', outFolder)
	#3. # of numerical character
	plotDNSFeature(dns, 'num', 'Percentage of Flows', 'Num. of Numerical Chars in the Domain', outFolder)
	#4. # of wildcards or periods
	plotDNSFeature(dns, 'nonnum', 'Percentage of Flows', 'Num. of Non-AlphaNum Chars in the Domain', outFolder)
	#5. # of IP addresses
	plotDNSFeature(dns, 'ipCount', 'Percentage of Flows', 'Num. of IPs per DNS request', outFolder)
	#6. ttl
	itemDict = {}
	itemList = []
	total = 0
	for tup in dns.keys():
		count = dns[tup]['count']
		total += count
		for item in dns[tup]['ttls']:
			try:
				itemDict[item] += count
			except KeyError:
				itemDict[item] = count
	for item in itemDict.keys():
		itemList.append( (item, itemDict[item]/float(total)*100) )
	itemList.sort(key=lambda x: x[1], reverse=True)
	if len(itemList) > 15:
		itemList = itemList[0:14]
	plt.clf()
	plt.bar(range(0, len(itemList)), [x[1] for x in itemList], align='center', alpha=0.5)
	plt.xticks(range(0, len(itemList)), [x[0] for x in itemList], rotation=45)
	plt.ylabel('Percentage of Flows')
	plt.title('DNS TTL Values')
	plt.savefig(outFolder+"ttls.pdf")
	#7. Alexa rank
	plotDNSFeature(dns, 'rank', 'Percentage of Flows', 'Alexa Top-N Lists', outFolder)
	

def main():
	global alexaMap
	parser = argparse.ArgumentParser(description="Probability Distribution of DNS Features in Dataset", add_help=True)
	parser.add_argument('-i', '--input', action="store", help="The input folder containing files generated by Joy")
	parser.add_argument('-j', '--json', action="store_true", default=False, help="Generate JSON output file")
	parser.add_argument('-f', '--figure', action="store_true", default=False, help="Generate Probability Distribution Figures")
	parser.add_argument('-a', '--allFile', action="store_true", default=False, help="Indicate whether treat all the file as together or separate")
	args = parser.parse_args()

	#setup input folder and output folders
	if args.input == None or not os.path.isdir(args.input):
		print("No valid input folder!") #verbose
		return
	else:
		joyFolder = args.input
		if not joyFolder.endswith('/'):
			joyFolder += '/'
	parentFolder = os.path.abspath(os.path.join(joyFolder, os.pardir))
	if not parentFolder.endswith('/'):
		parentFolder += '/'
	DNS_JSON_Folder = "%sDNS_JSON/" % (parentFolder)
	DNS_Figure_Folder = "%sDNS_Figure/" % (parentFolder) 
	ALEXA_Folder = "%sALEXA_JSON/" % (parentFolder)
	if not os.path.exists(DNS_JSON_Folder):
		os.mkdir(DNS_JSON_Folder)
	if args.figure:
		if not os.path.exists(DNS_Figure_Folder):
			os.mkdir(DNS_Figure_Folder)
	if not os.path.exists(ALEXA_Folder):
		os.mkdir(ALEXA_Folder)
	alexaFileName = "%salexa.json" % (ALEXA_Folder)
	if os.path.exists(alexaFileName):
		try:
			with open(alexaFileName, 'r') as fp:
				alexaMap = json.load(fp)
		except:
			pass

	#check if output JSON
	if args.json:
		if args.allFile == True: 
			dns = defaultdict()
			allFileName = ""
		files = os.listdir(joyFolder)
		for item in files:
			if args.allFile == True: 
				allFileName += (item.split('.'))[0] + "-"
			try:
				if args.allFile == False: 
					dns = defaultdict()
				ProcessDNS(joyFolder, item, dns)
				if args.allFile == False: 
					saveToJson(DNS_JSON_Folder, item, dns)
				if args.figure:
					if args.allFile == False: 
						plotDNS(dns, joyFolder, item, DNS_Figure_Folder)
			except:
				continue
		if args.allFile == True:
			allFileName +=  ".json"
			saveToJson(DNS_JSON_Folder, allFileName, dns)
			if args.figure:
				plotDNS(dns, joyFolder, allFileName, DNS_Figure_Folder)
	
	#check if output figures
	elif args.figure:
		if args.allFile == True: 
			allFileName = ""
			files = os.listdir(joyFolder)
			for item in files:
				allFileName += (item.split('.'))[0] + "-"
			allFileName +=  ".json"
			fName = DNS_JSON_Folder + (allFileName.split('.'))[0] + "_DNS.json"
			if os.path.exists(fName):
				try:
					with open(fName,'r') as fp:
						dns = json.load(fp)
						plotDNS(dns, joyFolder, allFileName, DNS_Figure_Folder)
				except:
					pass
			else:
				dns = defaultdict()
				for item in files:
					try:
						ProcessDNS(joyFolder, item, dns)
					except:
						continue
				try:
					plotDNS(dns, joyFolder, allFileName, DNS_Figure_Folder)
				except:
					pass
		else:
			files = os.listdir(joyFolder)
			for item in files:
				try:
					fName = DNS_JSON_Folder + (item.split('.'))[0] + "_DNS.json"
					if os.path.exists(fName):
						with open(fName,'r') as fp:
							dns = json.load(fp)
							plotDNS(dns, joyFolder, item, DNS_Figure_Folder)	
					else:
						dns = defaultdict()
						ProcessDNS(joyFolder, item, dns)
						plotDNS(dns, joyFolder, item, DNS_Figure_Folder)
				except:
					continue

	#save the alexaMap into JSON file
	try:
		json.dump(alexaMap, open(alexaFileName, 'w'))
	except:
		pass

if __name__ == "__main__":
	main()
	
