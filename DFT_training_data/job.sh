#!/bin/bash
#SBATCH --job-name=ZnSb          # Job name
#SBATCH --cpus-per-task=1        # Number of cores per MPI rank 
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=56     # How many tasks on each node
#SBATCH --output=log.out         # Standard output and error log

srun -n $SLURM_NTASKS --mpi=pmi2 ~/bin/vasp_std > std.out 
#mpiexec -n $SLURM_NTASKS ~/bin/vasp_std_5.4.1 > std.out 
