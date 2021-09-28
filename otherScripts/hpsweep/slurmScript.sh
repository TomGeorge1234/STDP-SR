#!/bin/bash 
#SBATCH --job-name=hpsweep              #name of the job to find when calling >>>sacct or >>>squeue 
#SBATCH --ntasks=256                  #how many independent script you are hoping to run 
#SBATCH --time=18:00:00                         #compute time 
#SBATCH --mem-per-cpu=6000MB 
#SBATCH --cpus-per-task=1 
#SBATCH --output=./logs/%j.log                  #where to save output log files (julia script prints here) 
#SBATCH --error=./logs/%j.err                   #where to save output error files 
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.600000 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.601569 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.603137 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.604706 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.606275 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.607843 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.609412 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.610980 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.612549 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.614118 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.615686 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.617255 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.618824 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.620392 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.621961 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.623529 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.625098 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.626667 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.628235 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.629804 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.631373 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.632941 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.634510 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.636078 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.637647 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.639216 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.640784 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.642353 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.643922 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.645490 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.647059 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.648627 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.650196 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.651765 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.653333 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.654902 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.656471 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.658039 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.659608 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.661176 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.662745 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.664314 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.665882 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.667451 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.669020 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.670588 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.672157 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.673725 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.675294 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.676863 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.678431 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.680000 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.681569 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.683137 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.684706 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.686275 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.687843 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.689412 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.690980 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.692549 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.694118 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.695686 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.697255 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.698824 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.700392 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.701961 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.703529 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.705098 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.706667 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.708235 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.709804 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.711373 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.712941 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.714510 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.716078 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.717647 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.719216 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.720784 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.722353 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.723922 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.725490 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.727059 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.728627 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.730196 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.731765 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.733333 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.734902 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.736471 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.738039 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.739608 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.741176 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.742745 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.744314 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.745882 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.747451 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.749020 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.750588 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.752157 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.753725 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.755294 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.756863 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.758431 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.760000 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.761569 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.763137 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.764706 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.766275 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.767843 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.769412 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.770980 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.772549 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.774118 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.775686 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.777255 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.778824 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.780392 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.781961 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.783529 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.785098 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.786667 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.788235 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.789804 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.791373 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.792941 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.794510 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.796078 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.797647 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.799216 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.800784 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.802353 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.803922 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.805490 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.807059 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.808627 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.810196 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.811765 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.813333 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.814902 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.816471 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.818039 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.819608 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.821176 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.822745 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.824314 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.825882 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.827451 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.829020 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.830588 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.832157 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.833725 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.835294 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.836863 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.838431 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.840000 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.841569 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.843137 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.844706 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.846275 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.847843 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.849412 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.850980 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.852549 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.854118 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.855686 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.857255 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.858824 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.860392 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.861961 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.863529 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.865098 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.866667 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.868235 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.869804 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.871373 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.872941 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.874510 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.876078 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.877647 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.879216 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.880784 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.882353 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.883922 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.885490 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.887059 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.888627 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.890196 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.891765 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.893333 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.894902 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.896471 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.898039 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.899608 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.901176 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.902745 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.904314 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.905882 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.907451 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.909020 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.910588 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.912157 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.913725 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.915294 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.916863 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.918431 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.920000 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.921569 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.923137 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.924706 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.926275 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.927843 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.929412 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.930980 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.932549 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.934118 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.935686 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.937255 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.938824 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.940392 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.941961 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.943529 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.945098 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.946667 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.948235 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.949804 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.951373 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.952941 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.954510 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.956078 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.957647 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.959216 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.960784 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.962353 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.963922 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.965490 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.967059 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.968627 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.970196 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.971765 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.973333 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.974902 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.976471 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.978039 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.979608 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.981176 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.982745 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.984314 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.985882 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.987451 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.989020 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.990588 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.992157 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.993725 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.995294 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.996863 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 0.998431 &
srun --ntasks=1 python clusterSweep.py 1.000000 0.020000 3.000000 0.800000 1.000000 &
wait