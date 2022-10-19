#!/bin/bash 
#SBATCH --job-name=hpsweep              #name of the job to find when calling >>>sacct or >>>squeue 
#SBATCH --ntasks=25                  #how many independent script you are hoping to run 
#SBATCH --time=18:00:00                         #compute time 
#SBATCH --mem-per-cpu=6000MB 
#SBATCH --cpus-per-task=1 
#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) 
#SBATCH --error=./logs/%j.err                   #where to save output error files 
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.100000 0.100000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.100000 0.300000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.100000 1.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.100000 3.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.100000 10.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.300000 0.100000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.300000 0.300000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.300000 1.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.300000 3.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.300000 10.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.500000 0.100000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.500000 0.300000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.500000 1.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.500000 3.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.500000 10.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.700000 0.100000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.700000 0.300000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.700000 1.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.700000 3.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.700000 10.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.900000 0.100000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.900000 0.300000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.900000 1.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.900000 3.000000 5.0 30.000000 &
srun --ntasks=1 --nodes=1 python clusterSweep.py 4.000000 0.020000 0.040000 -0.400000 0.900000 10.000000 5.0 30.000000 &
