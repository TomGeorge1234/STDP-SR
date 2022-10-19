#!/bin/bash

#SBATCH --job-name="twoRooms"             #name of the job to find when calling >>>sacct or >>>squeue 
#SBATCH --ntasks=2                  #how many independent script you are hoping to run 
#SBATCH --time=14-00:00:00                         #compute time 
#SBATCH --mem-per-cpu=8000MB 
#SBATCH --cpus-per-task=1 

#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) 
#SBATCH --error=./logs/%j.err                   #where to save output error files 

#SBATCH --mail-user=tom.george.20@ucl.ac.uk
#SBATCH --mail-type=ALL

srun --ntasks=1 --nodes=1 python clusterRun.py twoRooms_huge &
srun --ntasks=1 --nodes=1 python clusterRun.py twoRooms_unbiased_huge &
wait 
