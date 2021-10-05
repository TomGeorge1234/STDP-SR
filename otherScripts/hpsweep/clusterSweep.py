import sys 
sys.path.insert(-1,"../../")
from mazeAgentUtils import *



t_sr         = float(sys.argv[1])
t_stdp       = float(sys.argv[2])
t_stdp_asymm = float(sys.argv[3])
a_stdp_asymm = float(sys.argv[4])
f            = float(sys.argv[5])
k            = float(sys.argv[6])


N=50
xcen = np.linspace(0,5,N+1)[:-1]
xcen += (xcen[1] - xcen[0]) / 2
ycen = np.array([0.1]*N)
centres = np.vstack((xcen,ycen)).T

#Default parameters for MazeAgent 
params = { 

          #Maze params 
          'mazeType'            : 'loop',  #type of maze, define in getMaze() function
          'stateType'           : 'gaussianThreshold', #feature on which to TD learn (onehot, gaussian, gaussianCS, circles, bump)
          'movementPolicy'      : 'windowsScreensaver',  #movement policy (raudies, random walk, windows screensaver)
          'roomSize'            : 5,          #maze size scaling parameter, metres
          'dt'                  : 0.002,       #simulation time disretisation 
          'dx'                  : 0.01,       #space discretisation (for plotting, movement is continuous)
          'initPos'             : [0.1,0.1],  #initial position [x0, y0], metres
          'centres'             : centres,       #array of receptive field positions. Overwrites nCells
          'sigma'               : 1,          #basis cell width scale (irrelevant for onehots)
          'doorsClosed'         : False,       #whether doors are opened or closed in multicompartment maze

          #TD params 
          'tau'                 : t_sr,          #TD decay time, seconds
          
          #STDP params
          'tau_STDP'            : t_stdp,      #rate trace decays
          'tau_STDP_asymm'      : t_stdp_asymm,          # tau- = this * tau+ 
          'a_STDP_asymm'        : a_stdp_asymm,       #post-before-pre potentiation factor = this * pre-before-post

          #Theta precession params
          'precessFraction'     : f,        #fraction of 2pi the prefered phase moves through
          'kappa'               : k,          # von mises spread parameter

}

agent =  MazeAgent(params)
agent.runRat(trainTime=60)
plotter = Visualiser(agent)

fig, ax = plotter.plotMAveraged(time=10)
saveFigure(fig,'Mav',specialLocation='../../figures/clusterSweep/Mav_%g_%g_%g_%g_%g_%g.svg' %(int(1000*t_sr),int(1000*t_stdp),int(1000*t_stdp_asymm),int(1000*a_stdp_asymm),int(1000*f),int(1000*k)),figureDirectory="../../figures/")

fig1, ax1 = plotter.plotVarianceAndError()
saveFigure(fig1,'Mvar',specialLocation='../../figures/clusterSweep/Mvar_%g_%g_%g_%g_%g_%g.svg' %(int(1000*t_sr),int(1000*t_stdp),int(1000*t_stdp_asymm),int(1000*a_stdp_asymm),int(1000*f),int(1000*k)),figureDirectory="../../figures/")

R_10_w, R_10_nw, SNR_10_w, SNR_10_nw = agent.getMetrics(time=10)
R_60_w, R_60_nw, SNR_60_w, SNR_60_nw = agent.getMetrics(time=60)

data = [str(t_sr),str(t_stdp),str(t_stdp_asymm),str(a_stdp_asymm), str(f), str(k), str(round(R_10_w,5)), str(round(R_10_nw,5)), str(round(SNR_10_w,5)), str(round(SNR_10_nw,5)), str(round(R_60_w,5)), str(round(R_60_nw,5)), str(round(SNR_60_w,5)), str(round(SNR_60_nw,5))]
with open("sweepResults.txt", "a") as f: 
    if sum(1 for line in open('sweepResults.txt')) == 0:
        f.write("t_sr,t_stdp,t_stdp_asymm,a_stdp_asymm,f,k,R_10_w,R_10_nw,SNR_10_w,SNR_10_nw,R_60_w,R_60_nw,SNR_60_w,SNR_60_nw")
        f.write("\n")
    f.write(','.join(data))
    f.write('\n')
