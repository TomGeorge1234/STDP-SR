import subprocess
subprocess.run("rm slurmScript.sh", shell=True)

K = [0.5, 1, 2]
T = [20e-2, 25e-3, 30e-3]
A = [0.6, 0.7, 0.8]
F = [0.6, 0.8, 1]

pre_schpeel = [
"#!/bin/bash \n",
"#SBATCH --job-name=hpsweep              #name of the job to find when calling >>>sacct or >>>squeue \n",
"#SBATCH --nodes=4                               #number of nodes, i.e. computers, to request off the cluster (nodes typically have ~20 singled threaded cores) \n",
"#SBATCH --ntasks=%g                  #how many independent script you are hoping to run \n" %(len(K)*len(T)*len(A)*len(F)),
"#SBATCH --ntasks-per-node=25 \n",
"#SBATCH --time=18:00:00                         #compute time \n",
"#SBATCH --mem=8gb                               #memory to request \n",
"#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) \n",
"#SBATCH --error=./logs/%j.err                   #where to save output error files \n"
]


with open("slurmScript.sh","a") as new: 
    for line in pre_schpeel:
        new.write(line)

    for k in K:
        for t in T:
            for a in A: 
                for f in F:
                    new.write("python clusterSweep.py %f %f %f %f &" %(k, t, a, f))
                    new.write("\n")
