#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name=Rb_HgSn_Br
#SBATCH --cpus-per-task=1        # Number of cores per MPI rank 
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=56     # How many tasks on each node
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

export OMP_NUM_THREADS=1

#python random_searching_restart.py 0.cif K I Cd,Ge,Sn,Zn K_CdGeSnZn_I.pkl 1 K_CdGeSnZn_I.0.log > K_CdGeSnZn_I.1.log
#python random_searching_restart.py 0.cif Cs Br Cd,Ge,Hg Cs_CdGeHg_Br.pkl 0 > Cs_CdGeHg_Br.log
python ../random_searching_restart.py ../0.cif Rb Br Hg,Sn Rb_HgSn_Br.pkl 0 > Rb_HgSn_Br.log
