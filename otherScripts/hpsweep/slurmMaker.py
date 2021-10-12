import subprocess
import numpy as np 
subprocess.run("rm slurmScript.sh", shell=True)

T_SR = [4]
T_STDP_PLUS = [10e-3,20e-3,30e-3,50e-3,100e-3]
T_STDP_MINUS = [10e-3,20e-3,30e-3,50e-3,100e-3]
A_STDP = [-0.1,-0.2,-0.4,-0.6,-0.8,-1]
F = [0.5]
K = [1]
FR = [1,3,5,20,50]

n_tasks = len(T_SR)*len(T_STDP_PLUS)*len(T_STDP_MINUS)*len(A_STDP)*len(F)*len(K)#*len(FR)
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

    for t_sr in T_SR:
        for t_stdp_plus in T_STDP_PLUS:
            for t_stdp_minus in T_STDP_MINUS:
                for a_stdp in A_STDP:
                    for f in F:
                        for k in K: 
                            new.write("srun --ntasks=1 --nodes=1 python clusterSweep.py %f %f %f %f %f %f %s &" %(t_sr, t_stdp_plus, t_stdp_minus, a_stdp, f, k, str(FR).replace(" ","")))
                            new.write("\n")
    new.write("wait")

