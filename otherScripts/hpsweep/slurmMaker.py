import subprocess
import numpy as np 
subprocess.run("rm slurmScript.sh", shell=True)

K = [0.01,0.05,0.2,0.4,0.5,1]
T_STDP = [25e-3,30e-3]
T_SR = [4,5]
A = [0.9,0.95,0.98]
F = [0.6,0.7,0.8]

#K = [1]
#T_STDP = [20e-3]
#T_SR = [3]
#A = [0.8]
#F = list(np.linspace(0.6,1,256))
n_tasks = len(K)*len(T_STDP)*len(T_SR)*len(A)*len(F)
print("%g scripts total" %n_tasks)

pre_schpeel = [
"#!/bin/bash \n",
"#SBATCH --job-name=hpsweep              #name of the job to find when calling >>>sacct or >>>squeue \n",
"#SBATCH --ntasks=%g                  #how many independent script you are hoping to run \n" %n_tasks,
"#SBATCH --time=18:00:00                         #compute time \n",
"#SBATCH --mem-per-cpu=6000MB \n",
"#SBATCH --cpus-per-task=1 \n",
"#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) \n",
"#SBATCH --error=./logs/%j.err                   #where to save output error files \n"
]


with open("slurmScript.sh","a") as new: 
    for line in pre_schpeel:
        new.write(line)

    for k in K:
        for t_stdp in T_STDP:
            for t_sr in T_SR:
                for a in A: 
                    for f in F:
                        new.write("srun --ntasks=1 --nodes=1 python clusterSweep.py %f %f %f %f %f &" %(k, t_stdp, t_sr, a, f))
                        new.write("\n")
    new.write("wait")

