import subprocess
import numpy as np 
subprocess.run("rm slurmScript.sh", shell=True)

T_SR = [3,4,5,10]
T_STDP = [10e-3,20e-3,25e-3,30e-3]
T_STDP_ASYMM = [1,1.5,2,2.5]
A_STDP_ASYMM = [-0.8,-0.9,-1,-1.1]
F = [0.4,0.6,0.8,1]
K = [0.5,1,1.5,2]
FR = [1,5,20,50]


n_tasks = len(T_SR)*len(T_STDP)*len(T_STDP_ASYMM)*len(A_STDP_ASYMM)*len(F) #*len(K)*len(FR)
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
        for t_stdp in T_STDP:
            for t_stdp_asymm in T_STDP_ASYMM:
                for a_stdp_asymm in A_STDP_ASYMM:
                    for f in F:
                        new.write("srun --ntasks=1 --nodes=1 python clusterSweep.py %f %f %f %f %f %s %s &" %(t_sr, t_stdp, t_stdp_asymm, a_stdp_asymm, f, str(K).replace(" ",""), str(FR).replace(" ","")))
                        new.write("\n")
    new.write("wait")

