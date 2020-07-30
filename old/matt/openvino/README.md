# OpenVINO Implementation of NetML Project
The OpenVINO implementation of the NetML project for deep learning.

## Walkthrough Guide
1. Be sure to have completed the entirety of the DAAL NetML Walkthrough, as this will be required later in this guide.
2. First download [Intel Distribution of OpenVINO Toolkit](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux).
3. Switch directories to the location of the downloaded .tgz file.
4. Unpack the .tgz file using the following commmand: `tar -xvzf l_openvino_toolkit_p_<version>.tgz`
5. Change to the newly unpacked `l_openvino_toolkit_p_<version>` directory.
6. Install the prerequisite dependencies OpenVINO requires using the ready-made script.
    - `sudo -E ./install_openvino_dependencies.sh`
7. Choose the Installation style you would like:
    - GUI: `sudo ./install_GUI.sh`
    - cmd: `sudo ./install.sh`
8. Follow on screen instructions to install.
9. Go to the Model Optimizer *Install Prerequisites* Directory
    - `cd <INSTALL_DIR>/deployment_tools/model_optimizer/`
    - *INSTALL_DIR by default will be `/opt/intel/openvino`.*
10. Run the install prerequisites script required for TensorFlow. Take note of into which python these libraries are installed, most likely it will be a python3.5+.
    - `./install_prerequisites_tf.sh`
    - *NOTE: For the sake of simplicity, we use the default configuration of installing the Model Optimizer dependencies globally. If you would like to do a manual installation using a virtualenv, the documentation to do so is located [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Config_Model_Optimizer.html).*
11. In order to compile and run OpenVINO applications, we must update several environment variables.
    - Run following script script to temporarily set env variables: `source /opt/intel/openvino/bin/setupvars.sh`
    - Add this line to the end of your .bashrc to set your env variables on shell startup: `source /opt/intel/openvino/bin/setupvars.sh`
12. Run the following command to retrieve the absolute path for the environment python. Be sure to deactivate all conda envs using `conda deactivate`.
    - `which python`
    - This may potentially give the path to a default `/usr/bin/python`. You will want to ensure for the next step that you add the appropriate version designation found from the install prerequisites script e.g. `/usr/bin/python3.5`.
13. Using this absolute path, modify the shebang interpreter path in the `vinoInference.py` and `mo_tf.py` scripts. This ensures that the scripts will properly use the openVINO installed python upon running.
14. Activate the Anaconda Intel environment.
    - `conda activate idp`
15. Navigate to the directory containing all of the DAAL scripts.
    - `cd <PATH>/NetML/`
16. Using the pre-built Artificial Neural Network (ANN) from the DAAL walkthrough, we will construct an optimized OpenVINO ANN using OpenVINO's Model Optimizer and Inference Engine.
17. The "training" of the OpenVINO ANN entails transforming the current (trained) TensorFlow model into an optimized format that the OpenVINO Inference Engine can work with. To do this, we will use the same `daalClassifyWithTime.py` script used before, but with the input data file being your test.json file (to properly setup input dimensions) and the vinoANN spec for the model. The command will run the `mo_tf.py` script, which invokes the Model Optimizer for a TensorFlow model.
    - `python daalClassifyWithTime.py --workDir=pathGen --select=pathGen/test.json --classify --output=params.txt --model=vinoANN --http --tls`
18. You should now find in the specified working directory 4 new ANNmodel files. The xml and bin files contain the network information that will be provided to the Inference Engine, the mapping file contains meta information about the model, and the pb file contains the frozen TensorFlow model that was optimized by the Model Optimizer.
19. Once the network is successfully optimized, we can use the optimized model for inference on our test data.
    - `python daalClassifyWithTime.py --workDir=pathGen --select=pathGen/test.json --test --input=params.txt --model=vinoANN --http --tls`
20. The test dataset accuracy and inference time should be output, and steps 17-19 can be repeated for any new dataset.
