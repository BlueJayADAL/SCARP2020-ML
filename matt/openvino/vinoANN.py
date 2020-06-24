import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np
import time
import os

'''
	This script will act as an expansion of an already trained Tensorflow Keras
		ANN. It converts the Keras model from its native .hdf5 format into
		a .pb file that can then be used by OpenVINO.
'''

class vinoANN:
	'''
	## define globals
	# Initialize placeholders for data and labels
	x = tf.placeholder(dtype = tf.float32, shape = [None, None])
	y = tf.placeholder(dtype = tf.float32, shape = [None])

	# Define output layer (logit)
	logits = tf.contrib.layers.fully_connected(x, 1, tf.nn.relu)
	'''

	def __init__(self, numTLSFeature, numDNSFeature, numHTTPFeature, numTimesFeature, numLengthsFeature, numDistFeature):
		self.numTLSFeature = numTLSFeature
		self.numDNSFeature = numDNSFeature
		self.numHTTPFeature = numHTTPFeature
		self.numTimesFeature = numTimesFeature
		self.numLengthsFeature = numLengthsFeature
		self.numDistFeature = numDistFeature

	##OpenVINO Model Optimizer
	#Optimizes saved keras ANN 
	def train(self, data, label, dummy, workDir):
		#hard coded NN input size for VINO
		batch = len(data)
		dWidth = len(data[0])
		#create work dir path
		inH5 = "%sANNmodel.h5" % workDir
		#start model evaluation time
		startTime = time.time()
		#read in trained model (also automatically compiles)
		loadedModel = keras.models.load_model(inH5)
		print("Loaded model from disk")
		#Convert ANN model to binary .pb file and save
		frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in loadedModel.outputs])
		tf.train.write_graph(frozen_graph, workDir, "ANNmodel.pb", as_text=False)
		#Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
		generateCommand = "./mo_tf.py --input_model %sANNmodel.pb --input_shape [%d,%d] --output_dir %s" % (workDir, batch, dWidth, workDir)
		print(generateCommand)
		os.system(generateCommand)
		'''
		#make predictions
		predictResultTest = loadedModel.predict_classes(data)
		endTime = time.time()
		print("Test (Artificial Neural Network) elapsed in %.3f seconds" %(endTime - startTime))
		#assess accuracy
		correctTest = np.sum(label == predictResultTest.flatten())
		return float(correctTest)/len(label)*100
		'''
		

	#OpenVINO Inference Engine
	def test(self, data, label, workDir):
		#Save data and labels into files to be accessed by VINO Inference Engine
		label = np.array(label)
		dataPath = "%svinoTestData.npy" % (workDir)
		labelPath = "%svinoTestLabel.npy" % (workDir)
		startTime = time.time()
		np.save(dataPath, data)
		np.save(labelPath, label)
		endTime = time.time()
		print("Data and Labels saved to disk for inference.")
		print("Write elapsed in %.3f seconds" %(endTime - startTime))
		inferenceCommand = "./vinoInference.py --model %sANNmodel.xml --input %s --labels %s" % (workDir, dataPath, labelPath)
		print(inferenceCommand)
		os.system(inferenceCommand)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
		"""
		*Freezes the state of a session into a pruned computation graph.*
		Creates a new computation graph where variable nodes are replaced by
		constants taking their current value in the session. The new graph will be
		pruned so subgraphs that are not necessary to compute the requested
		outputs are removed.
		@param session The TensorFlow session to be frozen.
		@param keep_var_names A list of variable names that should not be frozen, or None to freeze all the variables in the graph.
		@param output_names Names of the relevant graph outputs.
		@param clear_devices Remove the device directives from the graph for better portability.
		@return The frozen graph definition.
		"""

		graph = session.graph
		with graph.as_default():
			freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
			output_names = output_names or []
			output_names += [v.op.name for v in tf.global_variables()]
			# Graph -> GraphDef ProtoBuf
			input_graph_def = graph.as_graph_def()
			if clear_devices:
				for node in input_graph_def.node:
					node.device = ""
			frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
			return frozen_graph
	
