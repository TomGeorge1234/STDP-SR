#!/bin/bash

#SBATCH --job-name=hpsweep   		#name of the job to find when calling >>>sacct or >>>squeue
#SBATCH --nodes=4				#number of nodes, i.e. computers, to request off the cluster (nodes typically have ~20 singled threaded cores)
#SBATCH --ntasks=100			#how many independent script you are hoping to run 
#SBATCH --time=18:00:00				#compute time
#SBATCH --mem=8gb				#memory to request
#SBATCH --output=./logs/%j.log			#where to save output log files (julia script prints here)  
#SBATCH --error=./logs/%j.err			#where to save output error files

pwd; hostname; date

