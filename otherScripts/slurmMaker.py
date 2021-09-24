
K = [0.5, 1, 2]
T = [20e-2, 25e-3, 30e-3]
A = [0.6, 0.7, 0.8]
F = [0.6, 0.8, 1]

with open("slurmDefault.sh","r") as default:
    with open("slurmScript.sh","a") as new: 
        for line in default:
            new.write(line)

        for k in K:
            for t in T:
                for a in A: 
                    for f in F:
                            new.write("python clusterSweep.py %f %f %f %f" %(k, t, a, f))
