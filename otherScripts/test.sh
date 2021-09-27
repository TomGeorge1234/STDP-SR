#!/bin/bash 
#SBATCH --job-name=test              #name of the job to find when calling >>>sacct or >>>squeue 
#SBATCH --ntasks=3                  #how many independent script you are hoping to run 
#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) 
#SBATCH --error=./logs/%j.err                   #where to save output error files

python test.py &
python test.py &
python test.py &
