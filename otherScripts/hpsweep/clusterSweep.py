import sys 
sys.path.insert(-1,"../../")
from mazeAgentUtils import *
import ast


t_sr         = ast.literal_eval(sys.argv[1])
t_stdp_plus  = ast.literal_eval(sys.argv[2])
t_stdp_minus = ast.literal_eval(sys.argv[3])
a_stdp       = ast.literal_eval(sys.argv[4])
f            = ast.literal_eval(sys.argv[5])
k            = ast.literal_eval(sys.argv[6])
fr           = ast.literal_eval(sys.argv[7])
traintime    = ast.literal_eval(sys.argv[8])

N=50
xcen = np.linspace(0,5,N+1)[:-1]
xcen += (xcen[1] - xcen[0]) / 2
ycen = np.array([0.1]*N)
centres = np.vstack((xcen,ycen)).T

#Default parameters for MazeAgent 

if type(t_sr) != list:
    t_sr = [t_sr]
if type(t_stdp_plus) != list:
    t_stdp_plus = [t_stdp_plus]
if type(t_stdp_minus) != list:
    t_stdp_minus = [t_stdp_minus]
if type(a_stdp) != list:
    a_stdp = [a_stdp]
if type(f) != list:
    f = [f]
if type(k) != list:
    k = [k]
if type(fr) != list:
    fr = [fr]



for t_sr_ in t_sr:
    for t_stdp_plus_ in t_stdp_plus:
        for t_stdp_minus_ in t_stdp_minus:
            for a_stdp_ in a_stdp:
                for f_ in f:
                    for k_ in k:
                        for fr_ in fr:

                            params = { 

                                #Maze params 
                                'mazeType'            : 'loop',  #type of maze, define in getMaze() function
                                'stateType'           : 'gaussianThreshold', #feature on which to TD learn (onehot, gaussian, gaussianCS, circles, bump)
                                'movementPolicy'      : 'windowsScreensaver',  #movement policy (raudies, random walk, windows screensaver)
                                'roomSize'            : 5,          #maze size scaling parameter, metres
                                'dt'                  : 0.01,       #simulation time disretisation 
                                'dx'                  : 0.02,       #space discretisation (for plotting, movement is continuous)
                                'initPos'             : [0.1,0.1],  #initial position [x0, y0], metres
                                'centres'             : centres,       #array of receptive field positions. Overwrites nCells
                                'sigma'               : 1,          #basis cell width scale (irrelevant for onehots)
                                'doorsClosed'         : False,       #whether doors are opened or closed in multicompartment maze

                                #TD params 
                                'tau'                 : t_sr_,          #TD decay time, seconds
                                'peakFiringRate'      : fr_,          #peak firing rate of a cell (middle of place field, preferred theta phase)
                                
                                #STDP params
                                'tau_STDP_plus'       : t_stdp_plus_, 
                                'tau_STDP_minus'      : t_stdp_minus_, 
                                'a_STDP'              : a_stdp_,       #post-before-pre potentiation factor = this * pre-before-post

                                #Theta precession params
                                'precessFraction'     : f_,        #fraction of 2pi the prefered phase moves through
                                'kappa'               : k_,          # von mises spread parameter
                                }

                            agent =  MazeAgent(params)
                            agent.runRat(trainTime=traintime)
                            plotter = Visualiser(agent)

                            fig, ax = plotter.plotMAveraged(time=30)
                            saveFigure(fig,'Mav',specialLocation='../../figures/clusterSweep/Mav_%g_%g_%g_%g_%g_%g_%g.svg' %(int(1000*t_sr_),int(1000*t_stdp_plus_),int(1000*t_stdp_minus_),int(1000*a_stdp_),int(1000*f_),int(1000*k_),int(1000*fr_)),figureDirectory="../../figures/")

                            fig1, ax1 = plotter.plotVarianceAndError()
                            saveFigure(fig1,'Mvar',specialLocation='../../figures/clusterSweep/Mvar_%g_%g_%g_%g_%g_%g_%g.svg' %(int(1000*t_sr_),int(1000*t_stdp_plus_),int(1000*t_stdp_minus_),int(1000*a_stdp_),int(1000*f_),int(1000*k_),int(1000*fr_)),figureDirectory="../../figures/")
                            
                            R_w, R_nw, SNR_w, SNR_nw, skew_w, skew_nw, skew_m = agent.getMetrics(time=30)

                            data = [str(t_sr_),str(t_stdp_plus_),str(t_stdp_minus_),str(a_stdp_), str(f_), str(k_), str(fr_), str(round(R_w,5)), str(round(R_nw,5)), str(round(SNR_w,5)), str(round(SNR_nw,5)), str(round(float(skew_w),5)), str(round(float(skew_nw),5)), str(round(float(skew_m),5))]
                            with open("sweepResults.txt", "a") as file: 
                                if sum(1 for line in open('sweepResults.txt')) == 0:
                                    file.write("t_sr,t_stdp_plus,t_stdp_minus,a_stdp,f,k,fr,R_w,R_nw,SNR_w,SNR_nw,skew_w,skew_nw,skew_m")
                                    file.write("\n")
                                file.write(','.join(data))
                                file.write('\n')
