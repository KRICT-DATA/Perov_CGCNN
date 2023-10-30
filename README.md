# Perov_CGCNN
This is the repository of code and data for paper "Machine learning-enabled chemical space exploration of all-inorganic perovskites for photovoltaics" (https://doi.org/10.21203/rs.3.rs-3315454/v1)

## **Prerequisites:**
This codes are based on Python3.
This codes require to install prerequistes of [CGCNN](https://github.com/txie-93/cgcnn).
In addition, this codes require:
* Linux operating system
* [Pandas](https://pandas.pydata.org/) with [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
* [Atomic Simulation Environment](https://databases.fysik.dtu.dk/ase/index.html)
* [tqdm](https://tqdm.github.io/)
* [Matplotlib](https://matplotlib.org/stable/)
* [mpltern](https://mpltern.readthedocs.io/en/latest/) (for ternary plot)

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/en/latest/index.html). After installing conda, run the following command to create a new environment named cgcnn and install all prerequisites:
```
conda upgrade conda
conda create -n cgcnn python=3
conda activate cgcnn
pip install scikit-learn torch pymatgen pandas openpyxl ase tqdm matplotlib mpltern
```
### **Optional:**
* Computer cluster nodes with workload manager program like slurm or pbs.

## **Explanation**
### **"Data"** directory
"Exploration results obtained by CGCNN.xlsx" file contains screening results for 41,400 ABX3 perovskite compositions. "Training data for CGCNN.xlsx" file contains DFT calculation results for 3,159 ABX3 perovskite compositions.

### **"CGCNN_training"** directory 
This directory contains codes and files necessary to train CGCNN models. See README file in the "CGCNN_training" directory for details.

### **"CGCNN_searching"** directory 
This directory contains codes and necessary files for chemical exploration of ABX3 perovskite using the trained CGCNN models.
"job_submit_example.sh" file shows example for using "random_searching_restart.py" code. The "random_searching_restart.py" code will create log and pickle files. The log file contains decomposition energy, the number of generated atomic configurations for each composition. The pickle file contains the lowest-energy atomic configuration for each composition, as a format of ASE (Atomic Simulation Environment) atoms. Using the generated pickle file, "bandgap_model_best.pth.tar", "bandtype_model_best.pth.tar", and "predict.py" files, you can predict bandgap and bandtype for the lowest-energy atomic configuration at each composition.
See README file in the "CGCNN_searching" directory for details.

### **"Codes_for_figures"** directory 
This directory contains codes and neccessary files for reproducing figures of the paper.

## **Authors**
This codes were primarily written by [Jin-Soo Kim](https://orcid.org/0000-0002-8230-8783) and [Juhwan Noh](https://scholar.google.co.kr/citations?hl=en&user=1FWcaAIAAAAJ) who were advised by Dr. [Jino Im](https://scholar.google.co.kr/citations?hl=en&user=b5Buk0MAAAAJ).
