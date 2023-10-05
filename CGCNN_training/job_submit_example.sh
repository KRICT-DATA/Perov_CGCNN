#!/bin/bash
#SBATCH --job-name=CGCNN_bandtype
#SBATCH --cpus-per-task=1        # Number of cores per MPI rank 
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=56
#SBATCH --output=log.out         # Standard output and error log
python ../main.py --epochs 100000 --task classification --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 ./cif_dir > std.out
