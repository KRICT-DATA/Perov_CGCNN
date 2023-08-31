# Perov_CGCNN
This is the repository of code and data for paper "Machine learning-enabled chemical space exploration of all-inorganic perovskites for photovoltaics"

In "Data" directory,	"Exploration results obtained by CGCNN.xlsx" file contains screening results for 41,400 ABX3 perovskite compositions. "Training data for CGCNN.xlsx" file contains DFT calculation results for 3,159 ABX3 perovskite compositions.

"CGCNN_training" directory contains "CIF" directory containing 3,159 ABX3 perovskite structures used for CGCNN training. There are three .csv files: "id_prop_for_bandgap.csv", "id_prop_for_bandtype.csv", and "id_prop_for_decomp.csv". If you want to train CGCNN model for predicting one of the three properties, please rename the corresponding .csv file to "id_prop.csv". For example, rename "id_prop_for_decomp.csv" to "id_prop.csv" if you want to train CGCNN model predicting decomposition energy. All codes for CGCNN are available in GitHub (https://github.com/txie-93/cgcnn). 

"CGCNN_searching" directory contains codes and necessary files for chemical exploration of ABX3 perovskite using the trained CGCNN models. "job_submit_example.sh" file shows example for using "random_searching_restart.py" code. The "random_searching_restart.py" code will create log and pickle files. The log file contains decomposition energy, the number of generated atomic configurations for each composition. The pickle file contains the lowest-energy atomic configuration for each composition, as a format of ASE (Atomic Simulation Environment) atoms. Using the generated pickle file, "bandgap_model_best.pth.tar", "bandtype_model_best.pth.tar", and "predict.py" files, you can predict bandgap and bandtype for the lowest-energy atomic configuration at each composition.

"Codes_for_figures" directory contains codes and data to reproduce figures in the manuscript. 
