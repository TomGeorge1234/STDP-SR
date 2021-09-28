#!/bin/bash 
#SBATCH --job-name=hpsweep              #name of the job to find when calling >>>sacct or >>>squeue 
#SBATCH --ntasks=216                  #how many independent script you are hoping to run 
#SBATCH --time=18:00:00                         #compute time 
#SBATCH --mem-per-cpu=6000MB 
#SBATCH --cpus-per-task=1 
#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) 
#SBATCH --error=./logs/%j.err                   #where to save output error files 
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 4.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 4.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 4.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 4.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 4.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 4.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 4.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 4.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 4.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 5.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 5.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 5.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 5.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 5.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 5.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 5.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 5.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 5.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 10.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 10.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 10.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 10.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 10.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 10.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 10.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 10.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.025000 10.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 4.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 4.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 4.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 4.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 4.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 4.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 4.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 4.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 4.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 5.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 5.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 5.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 5.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 5.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 5.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 5.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 5.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 5.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 10.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 10.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 10.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 10.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 10.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 10.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 10.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 10.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.100000 0.030000 10.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 4.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 4.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 4.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 4.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 4.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 4.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 4.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 4.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 4.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 5.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 5.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 5.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 5.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 5.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 5.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 5.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 5.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 5.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 10.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 10.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 10.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 10.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 10.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 10.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 10.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 10.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.025000 10.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 4.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 4.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 4.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 4.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 4.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 4.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 4.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 4.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 4.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 5.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 5.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 5.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 5.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 5.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 5.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 5.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 5.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 5.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 10.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 10.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 10.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 10.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 10.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 10.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 10.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 10.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.250000 0.030000 10.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 4.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 4.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 4.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 4.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 4.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 4.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 4.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 4.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 4.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 5.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 5.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 5.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 5.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 5.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 5.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 5.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 5.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 5.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 10.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 10.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 10.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 10.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 10.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 10.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 10.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 10.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.025000 10.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 4.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 4.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 4.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 4.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 4.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 4.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 4.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 4.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 4.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 5.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 5.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 5.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 5.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 5.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 5.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 5.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 5.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 5.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 10.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 10.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 10.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 10.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 10.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 10.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 10.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 10.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 0.500000 0.030000 10.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 4.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 4.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 4.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 4.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 4.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 4.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 4.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 4.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 4.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 5.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 5.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 5.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 5.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 5.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 5.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 5.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 5.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 5.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 10.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 10.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 10.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 10.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 10.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 10.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 10.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 10.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.025000 10.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 4.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 4.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 4.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 4.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 4.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 4.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 4.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 4.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 4.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 5.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 5.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 5.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 5.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 5.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 5.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 5.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 5.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 5.000000 0.990000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 10.000000 0.900000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 10.000000 0.900000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 10.000000 0.900000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 10.000000 0.950000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 10.000000 0.950000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 10.000000 0.950000 0.600000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 10.000000 0.990000 0.400000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 10.000000 0.990000 0.500000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 1.000000 0.030000 10.000000 0.990000 0.600000 &
wait