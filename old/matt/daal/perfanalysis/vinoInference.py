#!/home/derek/anaconda3/bin/python

import sys
import os
import numpy as np
from argparse import ArgumentParser, SUPPRESS
from openvino.inference_engine import IENetwork, IEPlugin

def build_argparser():
	parser = ArgumentParser(add_help=False)
	args = parser.add_argument_group('Options')
	args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
	args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True, type=str)
	args.add_argument("-i", "--input", help="Required. Path to a folder input file",required=True, type=str, nargs="+")
	args.add_argument("-d", "--device",help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
			"acceptable. The sample will look for a suitable plugin for device specified. Default value is CPU", default="CPU", type=str)
	args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
	args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)
	args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
	args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False, action="store_true")

	return parser
	
def main():
	args = build_argparser().parse_args()
	# Setup paths
	modelXML = args.model
	modelBin = os.path.splitext(modelXML)[0] + ".bin"
		
	# Read in data and labels
	data = np.load(args.input[0])
	labels = np.load(args.labels)	
	# Read IR
	net = IENetwork(model=modelXML, weights=modelBin)
	#print(net.inputs['dense_input'].shape)
	#print(net.outputs['dense_5/Sigmoid'].shape)
	plugin = IEPlugin(device = args.device)
	# Infer using Inference Engine
	execNet = plugin.load(network=net, num_requests=2)
	# Create predictions array and other loop vars
	#cutoff = data.shape[0]
	#predictions = np.empty(cutoff)
	#batch = 1000
	#batching = data[0:cutoff:batch]
	#print(batching[0].shape)
	
	#while index < endpoint:
		#batchData = data[lowEnd:highEnd]

	res = execNet.infer({'dense_input': data})
	correctTest = np.sum(labels == res['dense_5/Sigmoid'].flatten().round())
	testAccu = float(correctTest)/len(labels)*100
	print("VINO ANN Test Accuracy: %.3f%%" % (testAccu))

if __name__ == "__main__":
	main()

