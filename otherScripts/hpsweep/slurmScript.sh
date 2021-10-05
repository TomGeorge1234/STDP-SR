#!/bin/bash 
#SBATCH --job-name=hpsweep              #name of the job to find when calling >>>sacct or >>>squeue 
#SBATCH --ntasks=216                  #how many independent script you are hoping to run 
#SBATCH --time=18:00:00                         #compute time 
#SBATCH --mem-per-cpu=6000MB 
#SBATCH --cpus-per-task=1 
#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) 
#SBATCH --error=./logs/%j.err                   #where to save output error files 
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 1.500000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.000000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 2.500000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 1.500000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.000000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.030000 2.500000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 1.500000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.000000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.020000 2.500000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 1.500000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.000000 -0.500000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.400000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.400000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.400000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.400000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.400000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.400000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.450000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.450000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.450000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.450000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.450000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.450000 0.800000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.500000 0.600000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.500000 0.600000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.500000 0.600000 1.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.500000 0.800000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.500000 0.800000 1.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 5.000000 0.030000 2.500000 -0.500000 0.800000 1.500000 &
wait