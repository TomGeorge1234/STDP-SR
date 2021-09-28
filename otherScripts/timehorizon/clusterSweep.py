import sys 
sys.path.insert(-1,"../../")
from mazeAgentUtils import *


sigma = float(sys.argv[1])

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
          'sigma'               : sigma,                     #basis cell width scale (irrelevant for onehots)
          'doorsClosed'         : False,                 #whether doors are opened or closed in multicompartment maze
          
          'kappa'               : 1,          # von mises spread parameter
          'tau_STDP'            : 25e-3,      #rate trace decays
          'postpreAsymmetry'    : 0.8,        #depressionStrength = postpreAsymmetry * potentiationStrength 
          'precessFraction'     : 0.8,        #fraction of 2pi the prefered phase moves through

          'tau'                 : None   #sr timescale
          }


agent =  MazeAgent(params)
agent.runRat(trainTime=30,TDSRLearn=False,STDPLearn=True)
W_av, _ = agent.averageM(agent.W)
np.save("./results/W_sigma%g" %(int(1000*sigma)),W_av)

for tau in np.linspace(0.1,10,100):
    params['dt'] = 0.1
    params['tau'] = tau
    agent =  MazeAgent(params)
    agent.runRat(trainTime=30,TDSRLearn=True,STDPLearn=False)
    M_av, _ = agent.averageM(agent.M)
    np.save("./results/M_sigma%g_tau%g" %(int(1000*sigma),int(1000*tau)),W_av)
