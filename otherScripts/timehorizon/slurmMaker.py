import subprocess
subprocess.run("rm slurmScript.sh", shell=True)

S = np.linspace(0.25,2,8)
n_tasks = len(S)


pre_schpeel = [
"#!/bin/bash \n",
"#SBATCH --job-name=horizon              #name of the job to find when calling >>>sacct or >>>squeue \n",
"#SBATCH --ntasks=%g                  #how many independent script you are hoping to run \n" %n_tasks,
"#SBATCH --time=18:00:00                         #compute time \n",
"#SBATCH --mem-per-cpu=4000MB \n",
"#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) \n",
"#SBATCH --error=./logs/%j.err                   #where to save output error files \n"
]


with open("slurmScript.sh","a") as new: 
    for line in pre_schpeel:
        new.write(line)

    for s in S:
                    new.write("python clusterSweep.py %f &" %(s))
                    new.write("\n")
