import sys 
sys.path.insert(-1,"../")
from mazeAgentUtils import *

print("tester")

k = float(sys.argv[1])
t = float(sys.argv[2])
a = float(sys.argv[3])
f = float(sys.argv[4])



N=50
xcen = np.linspace(0,5,N+1)[:-1]
xcen += (xcen[1] - xcen[0]) / 2
ycen = np.array([0.1]*N)
centres = np.vstack((xcen,ycen)).T

#Default parameters for MazeAgent 
params = { 

          #Maze params 
          'mazeType'            : 'loop',                #type of maze, define in getMaze() function
          'stateType'           : 'gaussianThreshold',   #feature on which to TD learn (onehot, gaussian, gaussianCS, circles, bump)
          'movementPolicy'      : 'windowsScreensaver',  #movement policy (raudies, random walk, windows screensaver)
          'roomSize'            : 5,                     #maze size scaling parameter, metres
          'dt'                  : 0.001,                 #simulation time disretisation 
          'centres'             : centres,               #array of receptive field positions. Overwrites nCells
          'sigma'               : 1,                     #basis cell width scale (irrelevant for onehots)
          'doorsClosed'         : False,                 #whether doors are opened or closed in multicompartment maze
          'learnAllMatrices'    : True,                  #if True learns [STDP,TD] x [theta, noTheta] = all four. Other wise just STDPtheta and TDnoTheta 
          
          'kappa'               : k,          # von mises spread parameter
          'tau_STDP'            : t,      #rate trace decays
          'postpreAsymmetry'    : a,        #depressionStrength = postpreAsymmetry * potentiationStrength 
          'precessFraction'     : f,        #fraction of 2pi the prefered phase moves through


          }


agent =  MazeAgent(params)
agent.runRat(trainTime=30)
plotter = Visualiser(agent)

fig, ax, (R2, skill, area, L2)  = plotter.plotMAveraged()
saveFigure(fig,'Mav',specialLocation='../figures/paperFigures/figure2/2DE.svg')

data = [str(k),str(t),str(a), str(f), str(round(R2,5)), str(round(skill,5)), str(round(area,5)), str(round(L2,5))]
with open("sweepResults.txt", "a") as f: 
    f.write(','.join(data))
    f.write('\n')
