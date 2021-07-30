import numpy as np 

import pandas as pd 
from tqdm.notebook import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime 
import numbers
from pprint import pprint as pprint
import os
import scipy
from scipy.spatial import distance_matrix


import matplotlib
import matplotlib.pyplot as plt 
plt.style.use("seaborn")
rcParams['figure.dpi']= 300
rcParams['axes.labelsize']=5
rcParams['axes.labelpad']=2
rcParams['axes.titlepad']=3
rcParams['axes.titlesize']=5
rcParams['axes.xmargin']=0
rcParams['axes.ymargin']=0
rcParams['xtick.labelsize']=4
rcParams['ytick.labelsize']=4
rcParams['grid.linewidth']=0.5
rcParams['legend.fontsize']=4
rcParams['lines.linewidth']=0.5
rcParams.update({'figure.autolayout': True})
rcParams['xtick.major.pad']=2
rcParams['xtick.minor.pad']=2
rcParams['ytick.major.pad']=2
rcParams['ytick.minor.pad']=2
rcParams['xtick.color']='grey'
rcParams['ytick.color']='grey'
rcParams['figure.titlesize']='medium'
from cycler import cycler
rcParams['axes.prop_cycle']=cycler('color', ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'])
rcParams['image.cmap'] = 'inferno'

#Default parameters for MazeAgent 
defaultParams = { 
          'mazeType'           : 'oneRoom',  #type of maze, define in getMaze() function
          'stateType'          : 'gaussian', #feature on which to TD learn (onehot, gaussian, gaussianCS, fourier, circles)
          'movementPolicy'     : 'raudies',  #movement policy (raudies, random walk, windows screensaver)
          'roomSize'           : 1,          #maze size scaling parameter, metres
          'dt'                 : 0.01,       #simulation time disretisation 
          'dx'                 : 0.01,       #space discretisation (for plotting, movement is continuous)
          'tau'                : 2,          #TD decay time, seconds
          'TDdx'               : 0.02,       #rough distance between TD learning updates, metres 
          'alpha'              : 0.01,       #TD learning rate 
          'nCells'             : None,       #how many features to use
          'centres'            : None,       #array of receptive field positions. Overwrites nCells
          'speedScale'         : 0.16,       #movement speed scale, metres/second
          'rotSpeedScale'      : None,       #rotational speed scale, radians/second
          'initPos'            : None,       #initial position [x0, y0], metres
          'initDir'            : None,       #initial direction, unit vector
          'sigma'              : 0.3,        #feature cell width scale, relevant for  gaussin, gaussianCS, circles
          'doorsClosed'        : True,       #whether doors are opened or closed in multicompartment maze
          'successorFeatureNorm':1000       
}

class MazeAgent():
    """MazeAgent defines an agent moving around a maze. 
    The agent moves according to a predefined movement policy
    As the agent moves it learns a successor representation over state vectors according to a TD learning rule 
    The movement polcy is 
        (i)  continuous in space. There is no discretisation of location. Time is discretised into steps of dt
        (ii) completely decoupled from the TD learning.
    TD learning is 
        (i)  state general. i.e. it learns generic SRs for feature vectors which are not necessarily onehot. See de Cothi and Barry, 2020  
        (ii) time continuous. Defined in terms of a memory decay time tau, not unitless gamma. Any two staes can be used fro a TD learning step irrespective of their seperation in time. 
    As the rat moves and learns its position and time stamps are continually saved. Periodically a snapshot of the current SR matrix and state of other parameters in the maze are also saved. 
    """   
    def __init__(self,
                params={}):
        """Sets the parameters of the maze anad agent (using default if not provided) 
        and initialises everything. This includes: 
        •initilising history dataframes
        •making the maze (a dictionary of "walls" which cant be crossed)
        •setting position, velocity, time
        •discretising space into coordinates for later plotting
        •initialising basis features (gaussian centres, fourier frequencies etc.)
        •initialising SR matrix 

        Args:
            params (dict, optional): A dictionary of parameters which you want to differ from the default. Defaults to {}.
        """        
        print("Setting parameters")
        for key, value in defaultParams.items():
            setattr(self, key, value)
        self.updateParams(params)

        print("Initialising")
        self.initialise()

    def updateParams(self,
                     params : dict):        
        """Updates parameters from a dictionary. 
        All parameters found in params will be updated to new value

        Args:
            params (dict): dictionary of parameters to change
            initialise (bool, optional): [description]. Defaults to False.
        """        
        for key, value in params.items():
            setattr(self, key, value)

    def initialise(self): #should only be called once at the start 
        """Initialises the maze and agent. Should only be called once at the start.
        """        
        #initialise history dataframes
        print("   making state/history dataframes")
        self.mazeState = {}
        self.history = pd.DataFrame(columns = ['t','pos','delta','color','runID','firingRate','thetaPhase']) 
        self.snapshots = pd.DataFrame(columns = ['t','M','W','mazeState'])

        #set pos/vel
        print("   initialising velocity, position and direction")
        self.pos = self.initPos
        self.speed = self.speedScale
        self.dir = self.initDir

        #time and runID
        print("   setting time/run counters")
        self.t = 0
        self.runID = 0  
        self.thetaPhase = 10*(self.t%(1/10))*2*np.pi


        #make maze 
        print("   making the maze walls")
        self.walls = getWalls(mazeType=self.mazeType, roomSize=self.roomSize)
        self.mazeState['walls'] = self.walls

        #extent, xArray, yArray, discreteCoords
        print("   discretising position for later plotting")
        minx, maxx, miny, maxy = 0, 0, 0, 0
        for room in self.walls:
            wa = self.walls[room]
            minx, maxx, miny, maxy = min(minx,np.min(wa[...,0])), max(maxx,np.max(wa[...,0])), min(miny,np.min(wa[...,1])), max(maxy,np.max(wa[...,1])) 
        self.extent = np.array([minx,maxx,miny,maxy])
        self.width = maxx-minx
        self.height = maxy-miny
        self.xArray = np.arange(minx + self.dx/2, maxx, self.dx)
        self.yArray = np.arange(miny + self.dx/2, maxy, self.dx)[::-1]
        x_mesh, y_mesh = np.meshgrid(self.xArray,self.yArray)
        coordinate_mesh = np.array([x_mesh, y_mesh])
        self.discreteCoords = np.swapaxes(np.swapaxes(coordinate_mesh,0,1),1,2) #an array of discretised position coords over entire map extent 
        self.mazeState['extent'] = self.extent

        #handle None params
        print("   handling undefined parameters")
        if self.pos is None: 
            ex = self.extent
            self.pos = np.array([ex[0] + 0.2*(ex[1]-ex[0]),ex[2] + 0.2*(ex[3]-ex[2])])
        if self.dir is None: 
            if self.mazeType == 'longCorridor': self.dir = np.array([0,1])
            elif self.mazeType == 'loop': self.dir = np.array([1,0])
            else: self.dir = np.array([1,1]) / np.sqrt(2)
        if self.rotSpeedScale is None: 
            if self.mazeType == 'loop' or self.mazeType == 'longCorridor':
                self.rotSpeedScale = np.pi
            else: 
                self.rotSpeedScale = 3*np.pi
        if self.nCells is None: 
            ex = self.extent
            area, pcarea  = (ex[1]-ex[0])*(ex[3]-ex[2]), np.pi * ((self.sigma/2)**2)
            cellsPerArea = 10
            self.nCells = int(cellsPerArea * area / pcarea) #~10 in any given place
        if self.mazeType == 'TMaze':
            self.LRDecisionPending=True

        #initialise basis cells and M (successor matrix)
        print("   initialising basis features for learning")

        if self.stateType in ['gaussian', 'gaussianCS','gaussianThreshold', 'circles','onehot','bump']:
            if self.centres is not None: #if we don't provide locations for cell centres...
                self.nCells = self.centres.shape[0]
                self.stateSize = self.nCells
            else: #scatter some ourselves (making sure they aren't too close)
                print(self.nCells) 
                self.stateSize=self.nCells
                xcentres = np.random.uniform(self.extent[0],self.extent[1],self.nCells)
                ycentres = np.random.uniform(self.extent[2],self.extent[3],self.nCells)
                self.centres = np.array([xcentres,ycentres]).T
                inds = self.centres[:,0].argsort()
                self.centres = self.centres[inds]
                print("   checking basis cells aren't too close") 
                min_d = 0.1/0.9
                done = False
                while done != True:
                    min_d *= 0.9
                    print("     min seperation distance:  %.1f cm" %(min_d*100))
                    count = 0
                    while count <= 10:
                        d = distance_matrix(self.centres,self.centres)
                        d  += 0.1*np.eye(d.shape[0])
                        d_xid, d_yid = np.where(d < min_d)
                        print('      ',int(len(d_xid)/2),' overlapping pairs',end='\n')
                        if len(d_xid) == 0:
                            done = True 
                            break
                        to_remove = []
                        for i in range(len(d_xid)):
                            if d_xid[i] < d_yid[i]:
                                to_remove.append(d_xid[i])
                        to_remove = np.unique(to_remove)
                        xcentres = np.random.uniform(self.extent[0],self.extent[1],len(to_remove))
                        ycentres = np.random.uniform(self.extent[2],self.extent[3],len(to_remove))
                        self.centres[to_remove] = np.array([xcentres,ycentres]).T
                        count += 1
            self.M = np.eye(self.stateSize)
            self.W = self.M.copy()


            #order the place cells so successor matrix has some structure:
            if self.mazeType == 'twoRooms': #from centre outwards
                distance_to_centre = np.linalg.norm(np.array([1,0.5]) - self.centres,axis=1)
                distance_to_centre = distance_to_centre * (2*(self.centres[:,0]>1)-1)
                inds = distance_to_centre.argsort()
                self.centres = self.centres[inds]
            else: #from left to right
                inds = self.centres[:,0].argsort()
                self.centres = self.centres[inds]

        elif self.stateType == 'fourier':
            self.stateSize = self.nCells
            self.kVectors = np.random.rand(self.nCells,2) - 0.5
            self.kVectors /= np.linalg.norm(self.kVectors, axis=1)[:,None]
            self.kFreq = 2*np.pi / np.random.uniform(0.01,1,size=(self.nCells))
            self.phi = np.random.uniform(0,2*np.pi,size=(self.nCells))
            self.M = np.eye(self.stateSize)
            #self.M = np.zeros((self.stateSize,self.stateSize))
        
        if hasattr(self.sigma,"__len__"):
            if self.sigma.__len__() == self.nCells:
                self.sigmas = self.sigma
        else:
            self.sigmas = np.array([self.sigma]*self.nCells)

        #array of states, one for each discretised position coordinate 
        print("   calculating state vector at all discretised positions")
        self.discreteStates = self.positionArray_to_stateArray(self.discreteCoords,stateType=self.stateType) #an array of discretised position coords over entire map extent 
        
        #store time zero snapshot
        snapshot = pd.DataFrame({'t':[self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'mazeState':[self.mazeState]})
        self.snapshots = self.snapshots.append(snapshot)

        #STDP stuff
        print("   initialising STDP weight matrix and traces")
        self.W = np.zeros_like(self.M)
        self.postTrace = np.zeros(self.nCells) #causes depression 
        self.preTrace = np.zeros(self.nCells) #causes potentiation
        self.lastSpikeTime = -10

    def runRat(self,
            trainTime=10,
            plotColor=None,
            saveEvery=1,
            TDSRLearn=True,
            STDPLearn=True,
            TDlearnorder='distance'):
        """The main experiment call.
        A "run" consists of a period where the agent explores the maze according to the movement policy. 
        As it explores it learns, by TD, a successor representation over state vectors. 
        The can be called multiple times. Each successive run will be saved in self.history with an increasing runID
        Snapshots of the current SR matrix and mazeState can be saved along the way
        Runs can be interrupted with KeyboardInterrupt, data will still be saved. 
        Args:
            trainTime (int, optional): How long to explore. Defaults to 10.
            plotColor (str, optional): When plotting trajectory, what color to plot it. Defaults to 'C0'.
            saveEvery (int, optional): Frequency to save snapshots, in minutes. Defaults to 1.
            TDSRLearn (bool,optional): toggles whether to do TD learning 
            STDPLearn (bool, optional): toggles whether to do STDP learning 
        """        
        steps = int(trainTime * 60 / self.dt) #number of steps to perform 

        hist_t = np.zeros(steps)
        hist_pos = np.zeros((steps,2))
        hist_delta = np.zeros(steps)
        hist_plotColor = np.zeros(steps).astype(str)
        hist_runID = np.zeros(steps)
        hist_firingRate = np.zeros((steps,self.nCells))
        hist_thetaPhase = np.zeros(steps)

        try: 
            self.toggleDoors(self.doorsClosed) #confirm doors are open/closed if they should be 
        except KeyError: 
            pass 
        lastTDstep, distanceToTD = 0, np.random.exponential(self.TDdx) #2cm scale
        
        """Main training loop. Principally on each iteration: 
            • always updates motion policy
            • often does TD learning step

            • sometimes saves snapshot"""
        for i in tqdm(range(steps)): #main training loop 
            try:
                #update pos, velocity, direction and time according to movement policy
                self.movementPolicyUpdate()
                if i > 1:

                    # print(self.pos)
                    """STDP learning step"""
                    if (STDPLearn == True) and (self.stateType in  ['bump','gaussian', 'gaussianCS','gaussianThreshold', 'circles']):
                            hist_firingRate[i,:] = self.STDPLearningStep(dt = self.t - hist_t[i-1])

                    """TD learning step"""
                    if TDSRLearn == True: 
                        
                        alpha = self.alpha
                        try: alpha_ = alpha[0] * np.exp(-(i/steps)*(np.log(self.alpha[0]/self.alpha[1]))) #decaying alpha
                        except: alpha_ = self.alpha
                        

                        if TDlearnorder == 'distance':
                            if np.linalg.norm(self.pos - hist_pos[lastTDstep]) >= distanceToTD: #if it's moved over 2cm meters from last step 
                                dtTD = self.t - hist_t[lastTDstep]
                                delta = self.TDLearningStep(pos=self.pos, prevPos=hist_pos[lastTDstep], dt=dtTD, tau=self.tau, alpha=alpha_)
                                lastTDstep = i 
                                distanceToTD = np.random.exponential(self.TDdx)
                                hist_plotColor[i] = 'r'
                                hist_delta[i] = delta

                        elif TDlearnorder == 'decorrelated': 
                            id2 = np.random.randint(i)
                            learnDist = np.random.exponential(self.TDdx)
                            id1 = max(0,id2 - max(1,int((learnDist/self.speedScale) / self.dt)))
                            dtTD = hist_t[id2] - hist_t[id1]  
                            delta = self.TDLearningStep(pos=hist_pos[id2], prevPos=hist_pos[id1], dt=dtTD, tau=self.tau, alpha=alpha_)
                            hist_delta[i] = delta

                        elif TDlearnorder == 'everystep':
                            dtTD = self.t - hist_t[i-1]  
                            delta = self.TDLearningStep(pos=self.pos, prevPos=hist_pos[i-1], dt=dtTD, tau=self.tau, alpha=alpha_)
                            hist_delta[i] = delta

                self.thetaFrequency = 10
                self.thetaPhase = self.thetaFrequency*(self.t%(1/self.thetaFrequency))*2*np.pi #8Hz theta 

                #update history arrays
                hist_pos[i] = self.pos
                hist_t[i] = self.t
                hist_plotColor[i] = (plotColor or 'C'+str(self.runID))
                hist_runID[i] = self.runID
                hist_thetaPhase[i] = self.thetaPhase

                #save snapshot 
                if (isinstance(saveEvery, numbers.Number)) and (i % int(saveEvery * 60 / self.dt) == 0):
                    snapshot = pd.DataFrame({'t':[self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'mazeState':[self.mazeState]})
                    self.snapshots = self.snapshots.append(snapshot)
            except KeyboardInterrupt: 
                print("Keyboard Interrupt:")
                break
            except ValueError as error:
                print("ValueError:")
                print(error)
                print(f"   Rat position: {self.pos}")
                break

        self.runID += 1
        runHistory = pd.DataFrame({'t':list(hist_t[:i]), 'pos':list(hist_pos[:i]),'delta':list(hist_delta[:i]), 'color':list(hist_plotColor[:i]), 'runID':list(hist_runID[:i]), 'firingRate':list(hist_firingRate[:i]), 'thetaPhase':list(hist_thetaPhase[:i])})
        self.history = self.history.append(runHistory)
        snapshot = pd.DataFrame({'t': [self.t], 'M': [self.M.copy()], 'W': [self.W.copy()], 'mazeState':[self.mazeState]})
        self.snapshots = self.snapshots.append(snapshot)

        #find and save grid/place cells so you don't have to repeatedly calculate them when plotting 
        print("Calculating place and grid cells")
        self.gridFields = self.getGridFields(self.M)
        self.placeFields = self.getPlaceFields(self.M)

        if TDSRLearn == True: 
            # plotter = Visualiser(self)
            # plotter.plotTrajectory(starttime=(self.t/60)-0.2, endtime=self.t/60)
            delta = np.array(hist_delta)
            time = np.array(hist_t)
            time = time[delta!=0] / 60
            delta = delta[delta!=0]
            time, delta = time[::10], delta[::10]
            smooth_delta = [np.mean(delta[max(0,i-100):min(i+100,len(delta))]) for i in range(len(delta))]
            fig, ax = plt.subplots(figsize=(2,1))
            ax.scatter(time,delta,s=0.5,alpha=0.5)
            ax.scatter(time,smooth_delta,s=1,alpha=0.5,c='C2')
            ax.set_xlabel("Time / min")
            ax.set_ylabel("Update size")

    def TDLearningStep(self, pos, prevPos, dt, tau, alpha, reluM=False, lam=0.01):
        """TD learning step
            Improves estimate of SR matrix, M, by a TD learning step. 
            By default this is done using learning rule for generic feature vectors (see de Cothi and Barry 2020). 
            If stateType is onehot, additional efficiencies can be gained by using onehot specific learning rule (see Stachenfeld et al. 2017)
            Does time continuous TD learning (see Doya, 2000)
        Args:
            pos: position at t+dt (t)
            prevPos (array): position at t (t-dt)
            dt (float): time difference between two positions
            tau (float or int): memory decay time (analogous to gamma in TD, gamma = 1 - dt/tau)
            alpha (float): learning rate
            mask (bool or str): whether to mask TM update to update only cells near current location
            asynchronus (bool): update cells asynchronusly (like hopfield)
        """
        state = self.posToState(pos,stateType=self.stateType) 
        prevState = self.posToState(prevPos,stateType=self.stateType) 

        #onehot optimised TD learning 
        if self.stateType == 'onehot': 
            s_t = np.argwhere(prevState)[0][0]
            s_tplus1 = np.argwhere(state)[0][0]
            Delta = state + (tau / dt) * ((1 - dt/tau) * self.M[:,s_tplus1] - self.M[:,s_t])
            self.M[:,s_t] = self.M[:,s_t] + alpha * Delta - 2 * alpha * lam * self.M[:,s_t]

        #normal TD learning 
        else:
            delta = ((tau * dt) / (tau + dt)) * self.successorFeatureNorm * prevState + self.M @ ((tau/(tau + dt))*state - prevState)
            Delta = np.outer(delta, prevState)
            self.M = self.M + alpha * Delta - 2 * alpha * lam * self.M



        if reluM == True: 
            self.M = np.maximum(0,self.M)
        
        return np.linalg.norm(Delta)
                
    def STDPLearningStep(self,dt):       
        """Takes the curent theta phase and estimate firing rates for all basis cells according to a simple theta sweep model. 
           From here it samples spikes and performs STDP learning on a weight matrix.

        Args:
            dt (float): Time step length 

        Returns:
            float array: vector of firing rates for this time step 
        """   

        cellFiringRate = 10       #peak firing rate of a cell (middle of place field, preferred theta phase)
        tau_pre        = 20e-3      #rate potentiating trace decays
        tau_post       = 20e-3      #rate depressing trace decays 
        eta_pre        = 0.1       #learning rate for pre to post strengthening 
        eta_post       = 0.1       #learning rate for post to pre weakening
        a_pre          = 1        #per trace bump when cell spikes
        a_post         = 0.50        #post trace bump when cell spikes
        w_max          = 1       #max STDP weights
        w_min          = 0
        decayTime      = 10  
        sigmaTheta = np.pi/4
        baseline = 0

        thetaPhase = self.thetaPhase

        vectorToCells = self.vectorsToCellCentres(self.pos)
        sigmasToCellMidline = (np.dot(vectorToCells,self.dir) / np.linalg.norm(self.dir))  / self.sigmas #as mutiple of sigma
        preferedThetaPhase = np.pi + sigmasToCellMidline * (2/3) * np.pi
        peakFR = self.posToState(self.pos)
        phaseDiff = preferedThetaPhase - thetaPhase
        currentFR = peakFR * np.exp(-(phaseDiff)**2 / (2*sigmaTheta**2))
        currentFR = cellFiringRate*currentFR + baseline
        spike_list = []
        for cell in range(self.nCells):
            fr = currentFR[cell]
            n_spikes = np.random.poisson(fr*dt)
            if n_spikes != 0: 
                time_of_spikes = np.random.uniform(self.t,self.t+dt,n_spikes)
                for time in time_of_spikes:
                    spike_list.append([time,cell])
        spike_list = np.array(spike_list)
        if spike_list.shape[0] != 0: 
            spike_list = spike_list[np.argsort(spike_list[:,0])]
            lastSpikeTime = self.lastSpikeTime
            for i in range(len(spike_list)):
                time, cell = spike_list[i][0], int(spike_list[i][1])
                timeDiff = time - lastSpikeTime 
                self.postTrace = self.postTrace * np.exp(- timeDiff / tau_pre)
                self.preTrace = self.preTrace * np.exp(- timeDiff / tau_post)
                self.preTrace[cell] += a_pre
                self.postTrace[cell] += a_post
                weightsToPost = self.W[:,cell]
                weightsToPost += (w_max - weightsToPost) * eta_pre * self.preTrace
                # weightsToPost +=  eta_pre * self.preTrace
                weightsFromPost = self.W[cell,:]
                weightsFromPost += - weightsToPost * eta_post * self.postTrace
                weightsFromPost *= np.exp(-(time-lastSpikeTime)/decayTime)
                weightsToPost *= np.exp(-(time-lastSpikeTime)/decayTime)
                lastSpikeTime = time 


            self.lastSpikeTime=lastSpikeTime

        return currentFR

    def movementPolicyUpdate(self):
        """Movement policy update. 
            In principle this does a very simple thing: 
            • updates time by dt, 
            • updates position along the velocity direction 
            • updates velocity (speed and direction) accoridng to a movement policy
            In reality it's a complex function as the policy requires checking for immediate or upcoming collisions with all walls at each step.
            This is done by function self.checkWallIntercepts()
            What it does with this info (bounce off wall, turn to follow wall, etc.) depends on policy. 
        """
        dt = np.random.uniform(self.dt-0.01*self.dt,self.dt+0.01*self.dt)
        self.t += dt
        proposedNewPos = self.pos + self.speed * self.dir * dt
        proposedStep = np.array([self.pos,proposedNewPos])
        checkResult = self.checkWallIntercepts(proposedStep)
        if self.movementPolicy == 'randomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                randomTurnSpeed = np.random.normal(0,self.rotSpeedScale)
                self.dir = turn(self.dir,turnAngle=randomTurnSpeed*dt)
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.movementPolicy == 'trueRandomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                self.dir = turn(self.dir,turnAngle=np.random.uniform(0,2*np.pi))
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.movementPolicy == 'leftRightRandomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                self.dir = turn(self.dir,turnAngle=np.random.choice([0,np.pi]))
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.movementPolicy == 'raudies':
            if checkResult[0] == 'noImmediateCollision':
                self.pos = proposedNewPos
                self.speed = np.random.rayleigh(self.speedScale)
                randomTurnSpeed = np.random.normal(0,self.rotSpeedScale)
                self.dir = turn(self.dir,turnAngle=randomTurnSpeed*dt)
            if checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
            if checkResult[0] == 'collisionAhead':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'follow')
                randomTurnSpeed = np.random.normal(0,self.rotSpeedScale)
                self.dir = turn(self.dir, turnAngle=randomTurnSpeed*dt)

        if self.movementPolicy == 'windowsScreensaver':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.mazeType == 'loop':
            self.pos[0] = self.pos[0] % self.roomSize
        
        if self.mazeType == 'TMaze':
            if (self.pos[0] > self.roomSize+0.05) and (self.LRDecisionPending==True):
                if np.random.choice([0,1],p=[0.66,0.34]) == 0:
                    self.dir = np.array([0,1])
                else:
                    self.dir = np.array([0,-1])
                self.LRDecisionPending=False
            if self.pos[1] > self.extent[3] or self.pos[1] < self.extent[2]:
                self.pos = np.array([0,1])
                self.dir = np.array([1,0])
                self.LRDecisionPending=True


        #catchall instances a rat escapes the maze by accident, pops it 2cm within maze 
        if ((self.pos[0] < self.extent[0]) or 
            (self.pos[0] > self.extent[1]) or 
            (self.pos[1] < self.extent[2]) or 
            (self.pos[1] > self.extent[3])):
            print(self.pos)
            self.pos[0] = max(self.pos[0],self.extent[0]+0.02)
            self.pos[0] = min(self.pos[0],self.extent[1]-0.02)
            self.pos[1] = max(self.pos[1],self.extent[2]+0.02)
            self.pos[1] = min(self.pos[1],self.extent[3]-0.02)
            print("Rat escaped!")
            if self.mazeType == 'TMaze':
                self.dir=np.array([1,0])
                self.LRDecisionPending = True
            # plotter = Visualiser(self)
            # plotter.plotTrajectory(starttime=(self.t/60)-0.2, endtime=self.t/60)

    def vectorsToCellCentres(self,pos):
        """Takes a posisiton vector shape (2,) and returns an array of shape (nCells,2) of the 
        shortest distance to all cells, taking into account loop geometry etc. 

        Args:
            pos (array): position vector shape (2,)

        Returns:
            vectorToCells (array): shape (30,2)
        """        
        if self.mazeType == 'loop' and self.doorsClosed == False:
            pos_plus = pos + np.array([self.roomSize,0])
            pos_minus = pos - np.array([self.roomSize,0])
            positions = np.array([pos,pos_plus,pos_minus])
            vectors = self.centres[:,np.newaxis,:] - positions[np.newaxis,:,:]
            shortest = np.argmin(np.linalg.norm(vectors,axis=-1),axis=1)
            shortest_vectors = np.diagonal(vectors[:,shortest,:],axis1=0,axis2=1).T

        else:
            shortest_vectors = self.centres - self.pos
            

        return shortest_vectors

    def toggleDoors(self, doorsClosed = None): #this function could be made more advanced to toggle more maze options
        """Opens or closes door and updates mazeState
            mazeState stores the most recent version of the maze walls dictionary which will include 'door' wall only if doorsClosed is True
        Args:
            doorsClosed ([bool], optional): True is doors to be closed, False if doors to be opened. Defaults to None, in which case current door state is flipped.
        Returns:
            [dict]: the walls dictionary
        """        
        if doorsClosed is not None: 
            self.doorsClosed = doorsClosed
        else: self.doorsClosed = not self.doorsClosed

        walls = self.walls.copy()
        if self.doorsClosed == False: 
            del walls['doors']
            self.mazeState['walls'] = walls
        elif self.doorsClosed == True: 
            self.mazeState['walls'] = walls

        self.discreteStates = self.positionArray_to_stateArray(self.discreteCoords,stateType=self.stateType) #an array of discretised position coords over entire map extent 

        return self.mazeState['walls']

    def checkWallIntercepts(self,proposedStep): #proposedStep = [pos,proposedNextPos]
        """Given the cuurent proposed step [currentPos, nextPos] it calculates whether a collision with any of the walls exists along this step.
        There are three possibilities from most worrying to least: 
            • there is a collision ON the current step. Do something immediately. 
            • there is a collision along the current trajectory in the next few cm's, but not on the current step. Consider doing something. 
            • there is no collision coming up soon. Carry on as you are. 
        Args:
            proposedStep (array): The proposed step. np.array( [ [x_current, y_current] , [x_next, y_next] ] )

        Returns:
            tuple: (str, array), (<whether there is no collision, collision now or collision ahead> , <the wall in question>)
        """        
        s1, s2 = np.array(proposedStep[0]), np.array(proposedStep[1])
        pos = s1
        ds = s2 - s1
        stepLength = np.linalg.norm(ds)
        ds_perp = perp(ds)

        collisionList = [[],[]]
        futureCollisionList = [[],[]]

        #check if the current step results in a collision 
        walls = self.mazeState['walls'] #current wall state

        for wallObject in walls.keys():
            for wall in walls[wallObject]:
                w1, w2 = np.array(wall[0]), np.array(wall[1])
                dw = w2 - w1
                dw_perp = perp(dw)

                # calculates point of intercept between the line passing along the current step direction and the lines passing along the walls,
                # if this intercept lies on the current step and on the current wall (0 < lam_s < 1, 0 < lam_w < 1) this implies a "collision" 
                # if it lies ahead of the current step and on the current wall (lam_s > 1, 0 < lam_w < 1) then we should "veer" away from this wall
                # this occurs iff the solution to s1 + lam_s*(s2-s1) = w1 + lam_w*(w2 - w1) satisfies 0 <= lam_s & lam_w <= 1
                lam_s = (np.dot(w1, dw_perp) - np.dot(s1, dw_perp)) / (np.dot(ds, dw_perp)+1e-9)
                lam_w = (np.dot(s1, ds_perp) - np.dot(w1, ds_perp)) / (np.dot(dw, ds_perp)+1e-9)

                #there are two situations we need to worry about: 
                # • 0 < lam_s < 1 and 0 < lam_w < 1: the collision is ON the current proposed step . Do something immediately.
                # • lam_s > 1     and 0 < lam_w < 1: the collision is on the current trajectory, some time in the future. Maybe do something. 
                if (0 <= lam_s <= 1) and (0 <= lam_w <= 1):
                    collisionList[0].append(wall)
                    collisionList[1].append([lam_s,lam_w])
                    continue

                if (lam_s > 1) and (0 <= lam_w <= 1):
                    if lam_s * stepLength <= 0.03: #if the future collision is under 3cm away
                        futureCollisionList[0].append(wall)
                        futureCollisionList[1].append([lam_s,lam_w])
                        continue
        
        if len(collisionList[0]) != 0:
            wall_id = np.argmin(np.array(collisionList[1])[:,0]) #first wall you collide with on step 
            wall = collisionList[0][wall_id]
            return ('collisionNow', wall)
        
        elif len(futureCollisionList[0]) != 0:
            wall_id = np.argmin(np.array(futureCollisionList[1])[:,0]) #first wall you would collide with along current step 
            wall = futureCollisionList[0][wall_id]
            return ('collisionAhead', wall)
        
        else:
            return ('noImmediateCollision',None)

    def getPlaceFields(self, M=None):
        """Calculates receptive fiels of all place cells 
            There is one place cell for each feature cell. 
            A place cell (as  in de Cothi 2020) is defined as a thresholded linear combination of feature cells
            where the linear combination is a row of the SR matrix. 
        Args:
            M (array): SR matrix
        Returns:
            array: Receptive fields of shape [nCells, nX, nY]
        """        
        if M is None: 
            M = self.M
        
        #normalise: 
        M = M / np.diag(M)[:,np.newaxis]
        placeCellThreshold = 0.9       #place cell threshold value (fraction of its maximum)
        placeFields = np.einsum("ij,klj->ikl",M,self.discreteStates)
        threshold = placeCellThreshold*np.amax(placeFields,axis=(1,2))[:,None,None]
        threshold = placeCellThreshold
        placeFields = np.maximum(0,placeFields - threshold)
        return placeFields

    def getGridFields(self, M, alignToFinal=False):
        """Calculates receptive fiels of all grid cells 
            There is an equal number of grid cells as place cells and feature cells. 
            A grid cell (as in de Cothi 2020) is defined as a thresholded linear combination of feature cells
            where the linear combination weights are the eigenvectors of the SR matrix. 
        Args:
            M (array): SR matrix
            alignToFinal (bool): Since negative of eigenvec is also eigenvec try maximise overlap with final one (for making animations)
        Returns:
            array: Receptive fields of shape [nCells, nX, nY]
        """
        _, eigvecs = np.linalg.eig(M) #"v[:,i] is the eigenvector corresponding to the eigenvalue w[i]"
        eigvecs = np.real(eigvecs)
        gridCellThreshold = 0 
        gridFields = np.einsum("ij,kli->jkl",eigvecs,self.discreteStates)
        threshold = gridCellThreshold*np.amax(gridFields,axis=(1,2))[:,None,None]
        if alignToFinal == True:
            grids_final_flat = np.reshape(self.gridFields,(self.stateSize,-1))
            grids_flat = np.reshape(gridFields,(self.stateSize,-1))
            dotprods = np.empty(grids_flat.shape[0])
            for i in range(len(dotprods)):
                dotprodsigns = np.sign(np.diag(np.matmul(grids_final_flat,grids_flat.T)))
                gridFields *= dotprodsigns[:,None,None]
        gridFields = np.maximum(0,gridFields)
        return gridFields
    
    def posToState(self, pos, stateType=None, normalise=True, cheapNormalise=False): #pos is an [n1, n2, n3, ...., 2] array of 2D positions
        pos = np.array(pos)

        if stateType == None: stateType = self.stateType
    
        vector_to_cells = self.centres - pos
        distance_to_cells = [np.linalg.norm(vector_to_cells,axis=1)]
        closest_cell_ID = np.argmin(distance_to_cells)

        if (self.mazeType == 'loop') and (self.doorsClosed == False):
            distance_to_cells.append(np.linalg.norm(self.centres - pos + [self.extent[1],0],axis=1))
            distance_to_cells.append(np.linalg.norm(self.centres - pos - [self.extent[1],0],axis=1))
            
        if stateType == 'onehot':
            state = np.zeros(self.nCells)
            state[closest_cell_ID] = 1
        
        if stateType == 'gaussianThreshold':
            state = np.zeros(self.nCells)
            for distance in distance_to_cells: 
                state += np.maximum(np.exp(-distance**2 / (2*(self.sigmas**2))) - np.exp(-1/2) , 0) / (1-np.exp(-1/2))
        
        if stateType == 'gaussian':
            state = np.zeros(self.nCells)
            for distance in distance_to_cells: 
                state += np.exp(-distance**2 / (2*(self.sigmas**2)))

        if stateType == 'bump':
            state = np.zeros(self.nCells)
            for distance in distance_to_cells:
                state[distance<self.sigmas] += np.e * np.exp(-1/(1-(distance/self.sigmas)**2))[distance<self.sigmas]
                state[distance>=self.sigmas] += 0
            # state += 0.2



        if self.mazeType == 'twoRooms': 
            if pos[0]<1 and pos[1]>0.4 and pos[1]<0.6:
                cells_to_turn_off = (self.centres[:,0]>1)*np.invert((self.centres[:,1]>0.4)*(self.centres[:,1]<0.6))
            elif pos[0]>1 and pos[1]>0.4 and pos[1]<0.6:
                cells_to_turn_off = (self.centres[:,0]<1)*np.invert((self.centres[:,1]>0.4)*(self.centres[:,1]<0.6))
            elif pos[0]<1 and not(pos[1]>0.4 and pos[1]<0.6):
                cells_to_turn_off = (self.centres[:,0]>1)
            elif pos[0]>1 and not(pos[1]>0.4 and pos[1]<0.6):
                cells_to_turn_off = (self.centres[:,0]<1)
            turn_off_mask = np.invert(cells_to_turn_off)

            
            # if self.mazeType == 'twoRooms':
            #     turn_off_mask = np.ones(self.nCells)
            #     for i in range(len(self.centres)):
            #         line_of_sight_info = self.checkWallIntercepts(np.array([pos,self.centres[i]]))
            #         if line_of_sight_info[0] == 'collisionNow': 
            #             turn_off_mask[i] = 0 
            state *= turn_off_mask 
        
        return state


    def positionArray_to_stateArray(self, positionArray, stateType=None): 
        """Takes an array of 2D positions of size (n1, n2, n3, ..., 2)
        returns the state vector for each of these positions of size (n1, n2, n3, ..., N) where N is the size of the state vector
        Args:
            positionArray ([type]): [description]
            stateType ([type], optional): [description]. Defaults to None.
        """        
        if stateType == None: stateType = self.stateType
        states = np.zeros(positionArray.shape[:-1] + (self.nCells,))
        for idx in np.ndindex(positionArray.shape[:-1]):
            states[idx] = self.posToState(pos = positionArray[idx],stateType = stateType)
        
        return states
        
def getWalls(mazeType, roomSize=1):
    """Stores and returns dictionaries containing all the walls of a maze
    Args:
        mazeType (str): Name of the maze 
        roomSize (int, optional): scaling parameter for roomsize. Defaults to 1 metre.
    Returns:
        dict: wall dictionary
    """    
    walls = {}
    rs = roomSize
    if mazeType == 'oneRoom':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[rs,rs]],
                                [[rs,rs],[rs,0]],
                                [[rs,0],[0,0]]])
    elif mazeType == 'twoRooms':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[rs,rs]],
                                [[rs,rs],[rs,0.6*rs]],
                                [[rs,0.4*rs],[rs,0]],
                                [[rs,0],[0,0]]])
        walls['room2'] = np.array([
                                [[rs,0],[rs,0.4*rs]],
                                [[rs,0.6*rs],[rs,rs]],
                                [[rs,rs],[2*rs,rs]],
                                [[2*rs,rs],[2*rs,0]],
                                [[2*rs,0],[rs,0]]])
        walls['doors'] = np.array([[[rs,0.4*rs],[rs,0.6*rs]]])
    elif mazeType == 'fourRooms':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[0.4*rs,rs]],
                                [[0.6*rs,rs],[rs,rs]],
                                [[rs,rs],[rs,0.6*rs]],
                                [[rs,0.4*rs],[rs,0]],
                                [[rs,0],[0,0]]])
        walls['room2'] = np.array([
                                [[rs,0],[rs,0.4*rs]],
                                [[rs,0.6*rs],[rs,rs]],
                                [[rs,rs],[1.4*rs,rs]],
                                [[1.6*rs,rs],[2*rs,rs]],
                                [[2*rs,rs],[2*rs,0]],
                                [[2*rs,0],[rs,0]]])
        walls['room3'] = np.array([
                                [[0,rs],[0.4*rs,rs]],
                                [[0.6*rs,rs],[rs,rs]],
                                [[rs,rs],[rs,1.4*rs]],
                                [[rs,1.6*rs],[rs,2*rs]],
                                [[rs,2*rs],[0,2*rs]],
                                [[0,2*rs],[0,rs]]])
        walls['room4'] = np.array([
                                [[rs,rs],[1.4*rs,rs]],
                                [[1.6*rs,rs],[2*rs,rs]],
                                [[2*rs,rs],[2*rs,2*rs]],
                                [[2*rs,2*rs],[rs,2*rs]],
                                [[rs,2*rs],[rs,1.6*rs]],
                                [[rs,1.4*rs],[rs,rs]]])
        walls['doors'] = np.array([[[rs,0.4*rs],[rs,0.6*rs]],
                                        [[0.4*rs,rs],[0.6*rs,rs]],
                                        [[rs,1.4*rs],[rs,1.6*rs]],
                                        [[1.4*rs,rs],[1.6*rs,rs]]])
    elif mazeType == 'twoRoomPassage':
        walls['room1'] = np.array([
                                [[0,0],[rs,0]],
                                [[rs,0],[rs,rs]],
                                [[rs,rs],[0.75*rs,rs]],
                                [[0.25*rs,rs],[0,rs]],
                                [[0,rs],[0,0]]])
        walls['room2'] = np.array([
                                [[rs,0],[2*rs,0]],
                                [[2*rs,0],[2*rs,rs]],
                                [[2*rs,rs],[1.75*rs,rs]],
                                [[1.25*rs,rs],[rs,rs]],
                                [[rs,rs],[rs,0]]])
        walls['room3'] = np.array([
                                [[0,rs],[0,1.4*rs]],
                                [[0,1.4*rs],[2*rs,1.4*rs]],
                                [[2*rs,1.4*rs],[2*rs,rs]]])
        walls['doors'] = np.array([[[0.25*rs,rs],[0.75*rs,rs]],
                                [[1.25*rs,rs],[1.75*rs,rs]]])
    elif mazeType == 'longCorridor':
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[rs,rs]],
                                [[rs,rs],[rs,0]],
                                [[rs,0],[0,0]]])
        walls['longbarrier'] = np.array([
                                [[0.1*rs,0],[0.1*rs,0.9*rs]],
                                [[0.2*rs,rs],[0.2*rs,0.1*rs]],
                                [[0.3*rs,0],[0.3*rs,0.9*rs]],
                                [[0.4*rs,rs],[0.4*rs,0.1*rs]],
                                [[0.5*rs,0],[0.5*rs,0.9*rs]],
                                [[0.6*rs,rs],[0.6*rs,0.1*rs]],
                                [[0.7*rs,0],[0.7*rs,0.9*rs]],
                                [[0.8*rs,rs],[0.8*rs,0.1*rs]],
                                [[0.9*rs,0],[0.9*rs,0.9*rs]]])
    elif mazeType == 'rectangleRoom':
        ratio = np.pi/2.8
        walls['room1'] = np.array([
                                [[0,0],[0,rs]],
                                [[0,rs],[ratio*rs,rs]],
                                [[ratio*rs,rs],[ratio*rs,0]],
                                [[ratio*rs,0],[0,0]]])
    elif mazeType == 'loop':
        height = 0.2
        walls['room'] = np.array([
                                [[0,0],[rs,0]],
                                [[0,height],[rs,height]]])
        walls['doors'] = np.array([
                                [[0,0],[0,height]],
                                [[rs,0],[rs,height]]])
    
    elif mazeType == 'TMaze':
        walls['corridors'] = np.array([
                                [[0,0.05+rs],[rs,0.05+rs]],
                                [[0,-0.05+rs],[rs,-0.05+rs]],
                                [[rs,0.05+rs],[rs,rs+rs]],
                                [[rs+0.1,rs+rs],[rs+0.1,-rs+rs]],
                                [[rs,-0.05+rs],[rs,-rs+rs]]]) 
        walls['doors'] = np.array([[[rs,-0.05+rs-0.5],[rs+0.1,-0.05+rs-0.5]]])                               

    return walls

#MOVEMENT FUNCTIONS
def wallBounceOrFollow(currentDirection,wall,whatToDo='bounce'):
    """Given current direction, and wall and an instruction returns a new direction which is the result of implementing that instruction on the current direction
        wrt the wall. e.g. 'bounce' returns direction after elastic bounce off wall. 'follow' returns direction parallel to wall (closest to current heading)
    Args:
        currentDirection (array): the current direction vector
        wall (array): start and end coordinates of the wall
        whatToDo (str, optional): 'bounce' or 'follow'. Defaults to 'bounce'.
    Returns:
        array: new direction
    """    
    if whatToDo == 'bounce':
        wallPerp = perp(wall[1] - wall[0])
        if np.dot(wallPerp,currentDirection) <= 0:
            wallPerp = -wallPerp #it is now the perpendicular with smallest angle to dir 
        wallPar = wall[1] - wall[0]
        if np.dot(wallPar,currentDirection) <= 0:
            wallPar = -wallPar #it is now the parallel with smallest angle to dir 
        wallPar, wallPerp = wallPar/np.linalg.norm(wallPar), wallPerp/np.linalg.norm(wallPerp) #normalise
        dir_ = wallPar * np.dot(currentDirection,wallPar) - wallPerp * np.dot(currentDirection,wallPerp)
        newDir = dir_/np.linalg.norm(dir_)
    elif whatToDo == 'follow':
        wallPar = wall[1] - wall[0]
        if np.dot(wallPar,currentDirection) <= 0:
            wallPar = -wallPar #it is now the parallel with smallest angle to dir 
        wallPar = wallPar/np.linalg.norm(wallPar)
        dir_ = wallPar * np.dot(currentDirection,wallPar)
        newDir = dir_/np.linalg.norm(dir_)
    return newDir

def turn(currentDirection, turnAngle):
    """Turns the current direction by an amount theta, modulus 2pi
    Args:
        currentDirection (array): current direction 2-vector
        turnAngle (float): angle ot turn in radians
    Returns:
        array: new direction
    """    
    theta_ = theta(currentDirection)
    theta_ += turnAngle
    theta_ = np.mod(theta_, 2*np.pi)
    newDirection = np.array([np.cos(theta_),np.sin(theta_)])
    return newDirection

def perp(a=None):
    """Given 2-vector, a, returns its perpendicular
    Args:
        a (array, optional): 2-vector direction. Defaults to None.
    Returns:
        array: perpendicular to a
    """    
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0] 
    return b

def theta(segment):
    """Given a 'segment' (either 2x2 start and end positions or 2x1 direction bearing) 
         returns the 'angle' of this segment modulo 2pi
    Args:
        segment (array): The segment, (2,2) or (2,) array 
    Returns:
        float: angle of segment
    """    
    eps = 1e-6
    if segment.shape == (2,): 
        return np.mod(np.arctan2(segment[1],(segment[0] + eps)),2*np.pi)
    elif segment.shape == (2,2):
        return np.mod(np.arctan2((segment[1][1]-segment[0][1]),(segment[1][0] - segment[0][0] + eps)), 2*np.pi)

class Visualiser():
    def __init__(self, mazeAgent):
        self.mazeAgent = mazeAgent
        self.snapshots = mazeAgent.snapshots
        self.history = mazeAgent.history

    def plotMazeStructure(self,fig=None,ax=None,hist_id=-1,save=False):
        snapshot = self.snapshots.iloc[hist_id]
        extent, walls = snapshot['mazeState']['extent'], snapshot['mazeState']['walls']

        if (fig, ax) == (None, None): 
            fig, ax = plt.subplots(figsize=(4*(extent[1]-extent[0]),4*(extent[3]-extent[2])))
        for wallObject in walls.keys():
            for wall in walls[wallObject]:
                ax.plot([wall[0][0],wall[1][0]],[wall[0][1],wall[1][1]],color='darkgrey',linewidth=5)
            # ax.set_xlim(left=extent[0]-0.05,right=extent[1]+0.05)
            # ax.set_ylim(bottom=extent[2]-0.05,top=extent[3]+0.05)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('off')
        if save == True: 
            saveFigure(fig, 'mazeStructure')
        return fig, ax
    
    def plotTrajectory(self,fig=None, ax=None, hist_id=-1,starttime=0,endtime=2,skiprate=1,color=None):
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        startid = self.history['t'].sub(starttime*60).abs().to_numpy().argmin()
        endid = self.history['t'].sub(endtime*60).abs().to_numpy().argmin()
        trajectory = np.stack(self.history['pos'][startid:endid])[::skiprate]
        if color is None: 
            color = np.stack(self.history['color'][startid:endid])[::skiprate]
        ax.scatter(trajectory[:,0],trajectory[:,1],s=10,alpha=0.7,c=color,zorder=2)
        saveFigure(fig, "trajectory")
        return fig, ax

    
    def plotM(self,hist_id=-1, M=None,fig=None,ax=None,save=True,savename="",show=True,plotTimeStamp=False,colorbar=True,STDP=False):
        snapshot = self.snapshots.iloc[hist_id]
        if (ax is not None) and (fig is not None): 
            ax.clear()
        else:
            fig, ax = plt.subplots(figsize=(2,2))
        if M is None: 
            M = snapshot['M']
            if STDP==True: 
                M = snapshot['W'].T
        t = int(np.round(snapshot['t']))
        im = ax.imshow(M)
        divider = make_axes_locatable(ax)
        try: cax.clear()
        except: 
            pass
        if colorbar == True:
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('off')
        if save==True:
            saveFigure(fig, "M"+savename)
        if plotTimeStamp == True: 
            ax.text(100, 5,"%g s"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
        if show==False:
            plt.close(fig)
        return fig, ax

    def addTimestamp(self, fig, ax, i=-1):
        t = self.mazeAgent.saveHist[i]['t']
        ax.text(x=0, y=0, t="%.2f" %t)

    def plotPlaceField(self, hist_id=-1, time=None, fig=None, ax=None, number=None, show=True, animationCall=False, plotTimeStamp=False,save=True,STDP=False):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time).abs().to_numpy().argmin()
        #if a figure/ax objects are passed, clear the axis and replot the maze
        if (ax is not None) and (fig is not None): 
            ax.clear()
            self.plotMazeStructure(fig=fig, ax=ax, hist_id=hist_id)
        # else if they are not passed plot the maze
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        
        if number == None: number = np.random.randint(0,self.mazeAgent.stateSize-1)
        
        snapshot = self.snapshots.iloc[hist_id]
        M = snapshot['M']
        if STDP==True: 
            M = snapshot['W'].T
            print("using W")
        t = int(np.round(snapshot['t'] / 60))
        extent = snapshot['mazeState']['extent']
        placeFields = self.mazeAgent.getPlaceFields(M=M)
        ax.imshow(placeFields[number],extent=extent,interpolation=None)
        if plotTimeStamp == True: 
            ax.text(extent[1]-0.07, extent[3]-0.05,"%g"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
        if show==False:
            plt.close(fig)
        if save==True:
            saveFigure(fig, "placeField")
        return fig, ax
    
    def plotReceptiveField(self, number=None, hist_id=-1, fig=None, ax=None, show=True):
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        if number == None: number = np.random.randint(0,self.mazeAgent.nCells-1)
        extent = self.mazeAgent.extent
        rf = self.mazeAgent.discreteStates[..., number]
        ax.imshow(rf,extent=extent,interpolation=None)
        if show==False:
            plt.close(fig)
        saveFigure(fig, "receptiveField")
        return fig, ax
    

    def plotGridField(self, hist_id=-1, time=None, fig=None, ax=None, number=0, show=True, animationCall=False, plotTimeStamp=False,save=True,STDP=False):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()

        snapshot = self.snapshots.iloc[hist_id]
        M = snapshot['M']
        if STDP==True: 
            M = snapshot['W'].T    
        t = snapshot['t'] / 60
        extent = snapshot['mazeState']['extent']
        if hist_id == -1 and animationCall == False:
            gridFields = self.mazeAgent.gridFields
        else:
            gridFields = self.mazeAgent.getGridFields(M=M,alignToFinal=True)

        def sigmoid(x):
            return np.exp(x) / (np.exp(x) + np.exp(-x))
        if number == 'many': 
            fig = plt.figure(figsize=(10, 10*((extent[3]-extent[2])/(extent[1]-extent[0]))))
            gs = matplotlib.gridspec.GridSpec(6, 6, hspace=0.1, wspace=0.1)
            c=0
            # numberstoplot = np.array([60 + 5*i for i in np.arange(36)])
            numberstoplot = np.concatenate((np.array([0,1,2,3,4,5]),np.geomspace(6,gridFields.shape[0]-1,30).astype(int)))
            for i in range(6):
                for j in range(6):
                    ax = plt.subplot(gs[i,j])
                    # ax.imshow(sigmoid(gridFields[numberstoplot[c]]),extent=extent,interpolation=None)
                    ax.imshow(gridFields[numberstoplot[c]],extent=extent,interpolation=None)
                    ax.grid(False)
                    ax.axis('off')
                    ax.text(extent[1]-0.07, extent[3]-0.05,str(numberstoplot[c]+1),fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
                    c+=1

        else:
            #if a figure/ax objects are passed, clear the axis and replot the maze
            if (ax is not None) and (fig is not None): 
                ax.clear()
                self.plotMazeStructure(fig=fig, ax=ax, hist_id=hist_id)
            # else if they are not passed plot the maze
            if (fig, ax) == (None, None):
                fig, ax = self.plotMazeStructure(hist_id=hist_id)
            
            if number == None: number = np.random.randint(a=0,b=self.mazeAgent.stateSize-1)

            ax.imshow(gridFields[number],extent=extent,interpolation=None)

            if plotTimeStamp == True: 
                ax.text(extent[1]-0.07, extent[3]-0.05,"%g"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
            if show==False:
                plt.close(fig)
        
        if save==True:
            saveFigure(fig, "gridField")
        return fig, ax
        
    def plotFeatureCells(self, hist_id=-1,textlabel=True,shufflebeforeplot=False,centresOnly=False):
        fig, ax = self.plotMazeStructure(hist_id=hist_id)
        centres = self.mazeAgent.centres.copy()
        ids = np.arange(len(centres))
        if shufflebeforeplot==True:
            np.random.shuffle(ids)
        centres = centres[ids]
        for (i, centre) in enumerate(centres):
            # if i%10==0:

                if textlabel==True:
                    ax.text(centre[0],centre[1],str(ids[i]),fontsize=3,horizontalalignment='center',verticalalignment='center')
                if self.mazeAgent.mazeType == 'TMaze':
                    if abs(centre[1]-1)<0.001: color = 'C0'
                    elif centre[1]>1.001: color = 'C1'
                    elif centre[1]<0.999: color = 'C2'
                    else: color = 'C3'
                else:
                    color = 'C'+str(i)
                if centresOnly == True: 
                    if i == 30: c = 'r'
                    else: c = 'C0'
                    ax.plot(centre[0],centre[1],marker='x',markeredgewidth=3,color=c)
                else:
                    circle = matplotlib.patches.Ellipse((centre[0],centre[1]), 2*self.mazeAgent.sigmas[i], 2*self.mazeAgent.sigmas[i], alpha=0.5, facecolor=color)
                    ax.add_patch(circle)
                
        saveFigure(fig, "basis", tight_layout=False)
        return fig, ax 
    
    def plotHeatMap(self,smoothing=1):
        posdata = np.stack(self.mazeAgent.history['pos'])
        bins = [int(n/smoothing) for n in list(self.mazeAgent.discreteCoords.shape[:2])]
        bins.reverse()
        hist = np.histogram2d(posdata[:,0],posdata[:,1],bins=bins)[0]
        fig, ax = self.plotMazeStructure(hist_id=-1)
        ax.imshow(hist.T, extent=self.mazeAgent.extent)
        return fig, ax


    def animateField(self, number=0,field='place',interval=100):
        if field == 'place':
            fig, ax = self.plotPlaceField(hist_id=0,number=number,show=False,save=False)
            anim = FuncAnimation(fig, self.plotPlaceField, fargs=(None, fig, ax, number, False, True, True, False), frames=len(self.snapshots), repeat=False, interval=interval)
        elif field == 'grid':
            fig, ax = self.plotGridField(hist_id=0,number=number,show=False,save=False)
            anim = FuncAnimation(fig, self.plotGridField, fargs=(None, fig, ax, number, False, True, True, False), frames=len(self.snapshots), repeat=False, interval=interval)
        elif field == 'M':
            fig, ax = self.plotM(hist_id=0,show=False,save=False,colorbar=False)
            anim = FuncAnimation(fig, self.plotM, fargs=(None, fig, ax, False, "", False,True,False), frames=len(self.snapshots), repeat=False, interval=interval)
        
        today = datetime.strftime(datetime.now(),'%y%m%d')
        now = datetime.strftime(datetime.now(),'%H%M')
        saveFigure(anim,saveTitle=field+"Animation",anim=True)
        return anim

    def plotSTDPFiringRates(self,starttime=0,endtime=5/60):
        firingRates = np.array(list(self.mazeAgent.history['firingRate'].to_numpy()))
        t = self.mazeAgent.history['t'].to_numpy()
        theta = self.mazeAgent.history['thetaPhase'].to_numpy()
        startid = self.history['t'].sub(starttime*60).abs().to_numpy().argmin()
        endid = self.history['t'].sub(endtime*60).abs().to_numpy().argmin()
        fig, ax = plt.subplots(figsize=(5,1))
        for i in range(firingRates.shape[1]):
            ax.scatter(t[startid:endid],firingRates[startid:endid,i],s=1,alpha=0.5)
        ax.plot(t[startid:endid],0.05*np.sin(theta[startid:endid]),linewidth=0.5,c='black')
        ax.set_xlabel("Time / s")
        ax.set_ylabel("Cell firing rate")
        saveFigure(fig,"firingRate")
        return fig, ax 

def saveFigure(fig,saveTitle="",tight_layout=True,transparent=True,anim=False):
    """saves figure to file, by data (folder) and time (name) 
    Args:
        fig (matplotlib fig object): the figure to be saved
        saveTitle (str, optional): name to be saved as. Current time will be appended to this Defaults to "".
    """	

    today =  datetime.strftime(datetime.now(),'%y%m%d')
    if not os.path.isdir(f"../figures/{today}/"):
        os.mkdir(f"../figures/{today}/")
    figdir = f"../figures/{today}/"
    now = datetime.strftime(datetime.now(),'%H%M')
    path_ = f"{figdir}{saveTitle}_{now}"
    path = path_
    i=1
    while True:
        if os.path.isfile(path+".pdf") or os.path.isfile(path+".mp4"):
            path = path_+"_"+str(i)
            i+=1
        else: break
    if anim == True:
        fig.save(path + ".mp4")
    else:
        fig.savefig(path+".pdf", dpi=400,tight_layout=tight_layout,transparent=transparent)
    return path


