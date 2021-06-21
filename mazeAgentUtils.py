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
          'cellFiringRate'     : 10,         #peak firing rate of a cell (middle of place field, preferred theta phase)
          'centres'            : None,       #array of receptive field positions. Overwrites nCells
          'speedScale'         : 0.16,       #movement speed scale, metres/second
          'rotSpeedScale'      : None,       #rotational speed scale, radians/second
          'initPos'            : None,       #initial position [x0, y0], metres
          'initDir'            : None,       #initial direction, unit vector
          'sigma'              : 0.3,        #feature cell width scale, relevant for  gaussin, gaussianCS, circles
          'placeCellThreshold' : 0.5,        #place cell threshold value (fraction of its maximum)
          'gridCellThreshold'  : 0,          #grid cell threshold value (fraction of its maximum)
          'doorsClosed'        : True,       #whether doors are opened or closed in multicompartment maze
          'tau_pre'            : 20e-3,      #rate potentiating trace decays
          'tau_post'           : 20e-3,      #rate depressing trace decays 
          'eta_pre'            : 0.01,       #learning rate for pre to post strengthening 
          'eta_post'           : 0.01,       #learning rate for post to pre weakening
          'a_pre'              : 1,          #per trace bump when cell spikes
          'a_post'             : 0.3,        #post trace bump when cell spikes
          'w_max'              : 20e-3       #max STDP weights
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
        self.history = pd.DataFrame(columns = ['t','pos','color','runID','firingRate','thetaPhase']) 
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
        self.thetaPhase = 4*(self.t%(1/4))*2*np.pi


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
            if self.mazeType == 'longCorridor': self.dir = np.array([0.0001,1])
            elif self.mazeType == 'loop': self.dir = np.array([1,0.0001])
            else: self.dir = np.array([1,1]) / np.sqrt(2)
        if self.rotSpeedScale is None: 
            if self.mazeType == 'loop' or self.mazeType == 'longCorridor':
                self.rotSpeedScale = np.pi
            else: 
                self.rotSpeedScale = 3*np.pi
        if self.nCells is None: 
            ex = self.extent
            area, pcarea  = (ex[1]-ex[0])*(ex[3]-ex[2]), np.pi * ((self.sigma/2)**2)
            self.nCells = int(20 * area / pcarea) #~10 in any given place
            
        #initialise basis cells and M (successor matrix)
        print("   initialising basis features for learning")
        if self.stateType == 'onehot': 
            self.stateSize = len(self.xArray) * len(self.yArray)
            self.stateVec_asMatrix = np.zeros(shape=self.discreteCoords.shape[:-1])
            self.stateVec_asVector = self.stateVec_asMatrix.reshape((-1))
            self.M = np.eye(self.stateSize) 
            # self.M = np.zeros((self.stateSize,self.stateSize))
        elif self.stateType in ['gaussian', 'gaussianCS','gaussianThreshold', 'circles']:
            if self.centres is not None:
                self.nCells = self.centres.shape[0]
                self.stateSize = self.nCells
            else:
                self.stateSize=self.nCells
                xcentres = np.random.uniform(self.extent[0],self.extent[1],self.nCells)
                ycentres = np.random.uniform(self.extent[2],self.extent[3],self.nCells)
                self.centres = np.array([xcentres,ycentres]).T
            inds = self.centres[:,0].argsort()
            self.centres = self.centres[inds]
            self.M = np.eye(self.stateSize)
            # self.M = np.zeros((self.stateSize,self.stateSize))
            # self.M = np.ones((self.stateSize,self.stateSize)) / self.stateSize
        elif self.stateType == 'fourier':
            self.stateSize = self.nCells
            self.kVectors = np.random.rand(self.nCells,2) - 0.5
            self.kVectors /= np.linalg.norm(self.kVectors, axis=1)[:,None]
            self.kFreq = 2*np.pi / np.random.uniform(0.01,1,size=(self.nCells))
            self.phi = np.random.uniform(0,2*np.pi,size=(self.nCells))
            self.M = np.eye(self.stateSize)
            #self.M = np.zeros((self.stateSize,self.stateSize))
        
        self.sigmas = np.array([self.sigma]*self.nCells)
        # self.sigmas = np.random.uniform(self.sigma/3,2*self.sigma,size=(self.nCells))

        #array of states, one for each discretised position coordinate 
        print("   calculating state vector at all discretised positions")
        self.discreteStates = self.posToState(self.discreteCoords,stateType=self.stateType) #an array of discretised position coords over entire map extent 
        
        #store time zero snapshot
        snapshot = pd.DataFrame({'t':[self.t], 'M': [self.M.copy()], 'mazeState':[self.mazeState]})
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
        hist_plotColor = np.zeros(steps).astype(str)
        hist_runID = np.zeros(steps)
        hist_firingRate = np.zeros((steps,self.nCells))
        hist_thetaPhase = np.zeros(steps)

        self.toggleDoors(self.doorsClosed) #confirm doors are open/closed if they should be 
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
                    """STDP learning step"""
                    if (STDPLearn == True) and (self.stateType in  ['gaussian', 'gaussianCS','gaussianThreshold', 'circles']):
                            hist_firingRate[i,:] = self.STDPLearningStep(dt = self.t - hist_t[i-1])

                    """TD learning step"""
                    if TDSRLearn == True: 

                        alpha = self.alpha
                        try: alpha_ = alpha[0] * np.exp(-(i/steps)*(np.log(self.alpha[0]/self.alpha[1]))) #decaying alpha
                        except: alpha_ = self.alpha

                        if TDlearnorder == 'distance':
                            if np.linalg.norm(self.pos - hist_pos[lastTDstep]) >= distanceToTD: #if it's moved over 2cm meters from last step 
                                dtTD = self.t - hist_t[lastTDstep]
                                self.TDLearningStep(pos=self.pos, prevPos=hist_pos[lastTDstep], dt=dtTD, tau=self.tau, alpha=alpha_)
                                lastTDstep = i 
                                distanceToTD = np.random.exponential(self.TDdx)
                                hist_plotColor[i] = 'r'

                        elif TDlearnorder == 'decorrelated': 
                            id2 = np.random.randint(i)
                            learnDist = np.random.exponential(self.TDdx)
                            id1 = max(0,id2 - max(1,int((learnDist/self.speedScale) / self.dt)))
                            dtTD = hist_t[id2] - hist_t[id1]  
                            self.TDLearningStep(pos=hist_pos[id2], prevPos=hist_pos[id1], dt=dtTD, tau=self.tau, alpha=alpha_)

                        elif TDlearnorder == 'everystep':
                            dtTD = self.t - hist_t[i-1]  
                            self.TDLearningStep(pos=self.pos, prevPos=hist_pos[i-1], dt=dtTD, tau=self.tau, alpha=alpha_)

                self.thetaPhase = 4*(self.t%(1/4))*2*np.pi

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
            except ValueError:
                print("ValueError:")
                print(f"   Rat position: {self.pos}")
                break

        self.runID += 1
        runHistory = pd.DataFrame({'t':list(hist_t[:i]), 'pos':list(hist_pos[:i]), 'color':list(hist_plotColor[:i]), 'runID':list(hist_runID[:i]), 'firingRate':list(hist_firingRate[:i]), 'thetaPhase':list(hist_thetaPhase[:i])})
        self.history = self.history.append(runHistory)
        snapshot = pd.DataFrame({'t': [self.t], 'M': [self.M.copy()], 'W': [self.M.copy()], 'mazeState':[self.mazeState]})
        self.snapshots = self.snapshots.append(snapshot)

        #find and save grid/place cells so you don't have to repeatedly calculate them when plotting 
        print("Calculating place and grid cells")
        self.gridFields = self.getGridFields(self.M)
        self.placeFields = self.getPlaceFields(self.M)

        plotter = Visualiser(self)
        plotter.plotTrajectory(starttime=(self.t/60)-0.2, endtime=self.t/60)

    def TDLearningStep(self, pos, prevPos, dt, tau, alpha, mask=False, asynchronus=False, regularisation=0):
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
            delta = state + (tau / dt) * ((1 - dt/tau) * self.M[:,s_tplus1] - self.M[:,s_t])
            self.M[:,s_t] = self.M[:,s_t] + alpha * delta

        #normal TD learning 
        else:
            if asynchronus == False: 
                if mask != False: 
                    # alpha = alpha * np.exp(-2*np.linalg.norm(self.centres - self.pos,axis=1)/self.sigma)
                    alpha = alpha * np.exp(-np.linalg.norm(self.centres - self.pos,axis=1)**2/(2**(self.sigma/2)**2)).reshape((self.nCells,1))
                

                delta = prevState + (tau / dt) * (self.M @ ((1 - dt/tau)*state - prevState))
                self.M = (self.M +
                          alpha * (np.outer(delta, prevState) -
                          regularisation*(self.M**2)*self.M)
                         )

                
            elif asynchronus == True: 
                for i in np.random.permutation(self.nCells):
                    if mask is not False: 
                        alpha = alpha * np.exp(-np.linalg.norm(self.centres[i] - self.pos,axis=1)**2/(2**(self.sigma/2)**2))
                    delta = state + (tau / dt) * (self.M @ ((1 - dt/tau)*state - prevState))
                    self.M[i,:] = (self.M[i,:] +
                          alpha * np.outer(delta, state))[i,:]
        
            # equivalent to...
            # delta = prevState + (tau / dt) * (self.M @ (state - (1 + dt/tau)*prevState))
            # self.M = self.M + alpha * np.outer(delta, prevState)
            # are more general versions of...
            # delta = prevState + self.M @ ( 0.99 * state - prevState)
            # self.M = self.M + alpha * np.outer(delta, prevState)
    
    def STDPLearningStep(self,dt):       
        """Takes the curent theta phase and estimate firing rates for all basis cells according to a simple theta sweep model. 
           From here it samples spikes and performs STDP learning on a weight matrix.

        Args:
            dt (float): Time step length 

        Returns:
            float array: vector of firing rates for this time step 
        """        
        thetaPhase = self.thetaPhase

        vectorToCells = self.pos - self.centres
        alongPathDistToCellCentre = (np.dot(vectorToCells,self.dir) / np.linalg.norm(self.dir))  / self.sigmas #as mutiple of sigma
        preferedThetaPhase = np.pi - alongPathDistToCellCentre * (2/3) * np.pi
        peakFR = self.posToState(self.pos)
        sigmaTheta = np.pi/8
        phaseDiff = preferedThetaPhase - thetaPhase
        currentFR = peakFR * np.exp(-(phaseDiff)**2 / (2*sigmaTheta**2))

        spike_list = []
        for cell in range(self.nCells):
            fr = 20*currentFR[cell] + 0.5
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
                self.postTrace = self.postTrace * np.exp(- timeDiff / self.tau_pre)
                self.preTrace = self.preTrace * np.exp(- timeDiff / self.tau_post)
                self.preTrace[cell] += self.a_pre
                self.postTrace[cell] += self.a_post
                weightsToPost = self.W[:,cell]
                weightsToPost += (self.w_max - weightsToPost) * self.eta_pre * self.preTrace
                weightsFromPost = self.W[cell,:]
                weightsFromPost += - weightsToPost * self.eta_post * self.postTrace
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

        self.t += self.dt
        proposedNewPos = self.pos + self.speed * self.dir * self.dt
        proposedStep = np.array([self.pos,proposedNewPos])
        checkResult = self.checkWallIntercepts(proposedStep)

        if self.movementPolicy == 'randomWalk':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
                randomTurnSpeed = np.random.normal(0,self.rotSpeedScale)
                self.dir = turn(self.dir,turnAngle=randomTurnSpeed*self.dt)
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
                self.dir = turn(self.dir,turnAngle=randomTurnSpeed*self.dt)
            if checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
            if checkResult[0] == 'collisionAhead':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'follow')
                randomTurnSpeed = np.random.normal(0,self.rotSpeedScale)
                self.dir = turn(self.dir, turnAngle=randomTurnSpeed*self.dt)

        if self.movementPolicy == 'windowsScreensaver':
            if checkResult[0] != 'collisionNow': 
                self.pos = proposedNewPos
            elif checkResult[0] == 'collisionNow':
                wall = checkResult[1]
                self.dir = wallBounceOrFollow(self.dir,wall,'bounce')
        
        if self.mazeType == 'loop':
            self.pos[0] = self.pos[0] % self.roomSize

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
            # plotter = Visualiser(self)
            # plotter.plotTrajectory(starttime=(self.t/60)-0.2, endtime=self.t/60)


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

        self.discreteStates = self.posToState(self.discreteCoords,stateType=self.stateType) #an array of discretised position coords over entire map extent 

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
                lam_s = (np.dot(w1, dw_perp) - np.dot(s1, dw_perp)) / np.dot(ds, dw_perp)
                lam_w = (np.dot(s1, ds_perp) - np.dot(w1, ds_perp)) / np.dot(dw, ds_perp)

                #there are two situations we need to worry about: 
                # • 0 < lam_s < 1 and 0 < lam_w < 1: the collision is ON the current proposed step . Do something immediately.
                # • lam_s > 1     and 0 < lam_w < 1: the collision is on the current trajectory, some time in the future. Maybe do something. 
                if (0 <= lam_s <= 1) and (0 <= lam_w <= 1):
                    collisionList[0].append(wall)
                    collisionList[1].append([lam_s,lam_w])
                    continue

                if (lam_s > 1) and (0 <= lam_w <= 1):
                    if lam_s * stepLength <= 0.05: #if the future collision is under 3cm away
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

    def getPlaceFields(self, M):
        """Calculates receptive fiels of all place cells 
            There is one place cell for each feature cell. 
            A place cell (as  in de Cothi 2020) is defined as a thresholded linear combination of feature cells
            where the linear combination is a row of the SR matrix. 
        Args:
            M (array): SR matrix
        Returns:
            array: Receptive fields of shape [nCells, nX, nY]
        """        
        placeFields = np.einsum("ij,klj->ikl",M,self.discreteStates)
        threshold_ = self.placeCellThreshold
        threshold = threshold_*np.amax(placeFields,axis=(1,2))[:,None,None]
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
        gridFields = np.einsum("ij,kli->jkl",eigvecs,self.discreteStates)
        threshold_ = self.gridCellThreshold
        threshold = threshold_*np.amax(gridFields,axis=(1,2))[:,None,None]
        if alignToFinal == True:
            grids_final_flat = np.reshape(self.gridFields,(self.stateSize,-1))
            grids_flat = np.reshape(gridFields,(self.stateSize,-1))
            dotprods = np.empty(grids_flat.shape[0])
            for i in range(len(dotprods)):
                dotprodsigns = np.sign(np.diag(np.matmul(grids_final_flat,grids_flat.T)))
                gridFields *= dotprodsigns[:,None,None]
        gridFields = np.maximum(0,gridFields)
        return gridFields
    
    def posToState(self, pos, stateType=None, normalise=True): #pos is an [n1, n2, n3, ...., 2] array of 2D positions
        """Takes an array of 2D positions of size (n1, n2, n3, ..., 2)
        returns the state vector for each of these positions of size (n1, n2, n3, ..., N) where N is the size of the state vector
        Args:
            pos (array): array of positions, final dimension must be size 2
        Returns:
            array: array of states. Same shape as input except final dimension has gone from 2 to nCells.
        """ 
        if stateType == None: stateType = self.stateType

        if stateType == 'onehot':     
            x_s = pos[..., 0][...,None]
            y_s = pos[..., 1][...,None]
            x_s = np.tile(x_s, len(x_s.shape[:-1]) * (1,) + (len(self.xArray),))
            y_s = np.tile(y_s, len(y_s.shape[:-1]) * (1,) + (len(self.yArray),))
            x_s = x_s - self.xArray
            y_s = y_s - self.yArray
            x_s = np.argmin(np.abs(x_s),axis=-1)
            y_s = np.argmin(np.abs(y_s),axis=-1)
            onehotcoord = y_s * len(self.yArray) + x_s
            states = (np.arange(self.stateSize) == onehotcoord[...,None]).astype(int)
        
        if stateType in ['gaussian','gaussianCS','gaussianThreshold','circles']:
            centres = self.centres
            pos = np.expand_dims(pos,-2)
            diff = np.abs((centres - pos))
            dev = [np.linalg.norm(diff,axis=-1)]
            
            if (self.mazeType == 'loop') and (self.doorsClosed == False):
                diff_aroundloop = diff.copy()
                diff_aroundloop[..., 0] = (self.extent[1]-self.extent[0]) - diff_aroundloop[..., 0]
                dev_aroundloop = np.linalg.norm(diff_aroundloop,axis=-1)
                dev.append(dev_aroundloop)

            states = np.zeros_like(dev[0])

            for devs in dev:
                if stateType == 'circles':
                    states_ = devs
                    states_ = np.where(states > self.sigma1, 0, 1)
                elif stateType == 'gaussian': 
                    states_ = np.exp(-devs**2 / (2*self.sigmas**2))
                elif stateType == 'gaussianCS':
                    states_ = (np.exp(-devs**2 / (2*self.sigmas**2)) -  (1/2)*np.exp(-devs**2 / (2*(2*self.sigmas)**2)) )  / (1/2)
                elif stateType == 'gaussianThreshold':
                     states_ = np.maximum(np.exp(-devs**2 / (2*self.sigmas**2)) - np.exp(-1/2) , 0) / (1-np.exp(-1/2))
                states += states_

            states = states #all circle states st maximum FR = 1

        if stateType == 'fourier':
            phase = np.matmul(pos,self.kVectors.T) * self.kFreq + self.phi
            fr = np.cos(phase)
            states = fr 

        #normalise state 
        if normalise == True: 
            states = states / np.linalg.norm(states,axis=-1)[...,np.newaxis]

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
        height = 0.05
        walls['room'] = np.array([
                                [[0,0],[rs,0]],
                                [[0,height*rs],[rs,height*rs]]])
        walls['doors'] = np.array([
                                [[0,0],[0,height*rs]],
                                [[rs,0],[rs,height*rs]]])
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

    def plotMazeStructure(self,fig=None,ax=None,hist_id=-1):
        snapshot = self.snapshots.iloc[hist_id]
        extent, walls = snapshot['mazeState']['extent'], snapshot['mazeState']['walls']

        if (fig, ax) == (None, None): 
            fig, ax = plt.subplots(figsize=(4*(extent[1]-extent[0]),4*(extent[3]-extent[2])))
        for wallObject in walls.keys():
            for wall in walls[wallObject]:
                ax.plot([wall[0][0],wall[1][0]],[wall[0][1],wall[1][1]],color='darkgrey',linewidth=2)
            ax.set_xlim(left=extent[0]-0.05,right=extent[1]+0.05)
            ax.set_ylim(bottom=extent[2]-0.05,top=extent[3]+0.05)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('off')
        return fig, ax
    
    def plotTrajectory(self,fig=None, ax=None, hist_id=-1,starttime=0,endtime=2,skiprate=1):
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        startid = self.history['t'].sub(starttime*60).abs().to_numpy().argmin()
        endid = self.history['t'].sub(endtime*60).abs().to_numpy().argmin()
        trajectory = np.stack(self.history['pos'][startid:endid])[::skiprate]
        color = np.stack(self.history['color'][startid:endid])[::skiprate]
        ax.scatter(trajectory[:,0],trajectory[:,1],s=0.4,alpha=0.7,c=color,zorder=2)
        saveFigure(fig, "trajectory")
        return fig, ax

    
    def plotM(self,hist_id=-1, M=None,fig=None,ax=None,save=True,savename="",show=True,plotTimeStamp=False):
        snapshot = self.snapshots.iloc[hist_id]
        if (ax is not None) and (fig is not None): 
            ax.clear()
        else:
            fig, ax = plt.subplots(figsize=(2,2))
        if M is None: 
            M = snapshot['M']
        t = int(np.round(snapshot['t']))
        im = ax.imshow(M)
        divider = make_axes_locatable(ax)
        try: cax.clear()
        except: 
            pass
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

    def plotPlaceField(self, hist_id=-1, time=None, fig=None, ax=None, number=None, show=True, animationCall=False, plotTimeStamp=False,save=True):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time).abs().to_numpy().argmin()
        #if a figure/ax objects are passed, clear the axis and replot the maze
        if (ax is not None) and (fig is not None): 
            ax.clear()
            self.plotMazeStructure(fig=fig, ax=ax, hist_id=hist_id)
        # else if they are not passed plot the maze
        if (fig, ax) == (None, None):
            fig, ax = self.plotMazeStructure(hist_id=hist_id)
        
        if number == None: number = random.randint(a=0,b=self.mazeAgent.stateSize-1)
        
        snapshot = self.snapshots.iloc[hist_id]
        M = snapshot['M']
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
        if number == None: number = random.randint(a=0,b=self.mazeAgent.stateSize-1)
        extent = self.mazeAgent.extent
        rf = self.mazeAgent.discreteStates[..., number]
        ax.imshow(rf,extent=extent,interpolation=None)
        if show==False:
            plt.close(fig)
        saveFigure(fig, "receptiveField")
        return fig, ax
    

    def plotGridField(self, hist_id=-1, time=None, fig=None, ax=None, number=0, show=True, animationCall=False, plotTimeStamp=False,save=True):
        if time is not None: 
            hist_id = self.snapshots['t'].sub(time*60).abs().to_numpy().argmin()

        snapshot = self.snapshots.iloc[hist_id]
        M = snapshot['M']
        t = snapshot['t'] / 60
        extent = snapshot['mazeState']['extent']
        if hist_id == -1 and animationCall == False:
            gridFields = self.mazeAgent.gridFields
        else:
            gridFields = self.mazeAgent.getGridFields(M=M,alignToFinal=True)

        if number == 'many': 
            fig = plt.figure(figsize=(10, 10*((extent[3]-extent[2])/(extent[1]-extent[0]))))
            gs = matplotlib.gridspec.GridSpec(6, 6, hspace=0.1, wspace=0.1)
            c=0
            numberstoplot = np.concatenate((np.array([0,1,2,3,4,5]),np.geomspace(6,gridFields.shape[0]-1,30).astype(int)))
            for i in range(6):
                for j in range(6):
                    ax = plt.subplot(gs[i,j])
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
            
            if number == None: number = random.randint(a=0,b=self.mazeAgent.stateSize-1)

            ax.imshow(gridFields[number],extent=extent,interpolation=None)

            if plotTimeStamp == True: 
                ax.text(extent[1]-0.07, extent[3]-0.05,"%g"%t, fontsize=5,c='w',horizontalalignment='center',verticalalignment='center')
            if show==False:
                plt.close(fig)
        
        if save==True:
            saveFigure(fig, "gridField")
        return fig, ax
        
    def plotFeatureCells(self, hist_id=-1,textlabel=True,shufflebeforeplot=True):
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
                circle = matplotlib.patches.Ellipse((centre[0],centre[1]), 2*self.mazeAgent.sigmas[i], 2*self.mazeAgent.sigmas[i], alpha=0.1, facecolor= 'C'+str(i))
                ax.add_patch(circle)
        saveFigure(fig, "basis")
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
            fig, ax = self.plotM(hist_id=0,show=False,save=False)
            anim = FuncAnimation(fig, self.plotM, fargs=(None, fig, ax, False, "", False,True), frames=len(self.snapshots), repeat=False, interval=interval)
        
        today = datetime.strftime(datetime.now(),'%y%m%d')
        now = datetime.strftime(datetime.now(),'%H%M')
        saveFigure(anim,saveTitle=field+"Animation",anim=True)
        return anim


def saveFigure(fig,saveTitle="",tight_layout=True,transparent=True,anim=False):
    """saves figure to file, by data (folder) and time (name) 
    Args:
        fig (matplotlib fig object): the figure to be saved
        saveTitle (str, optional): name to be saved as. Current time will be appended to this Defaults to "".
    """	

    today =  datetime.strftime(datetime.now(),'%y%m%d')
    if not os.path.isdir(f"./figures/{today}/"):
        os.mkdir(f"./figures/{today}/")
    figdir = f"./figures/{today}/"
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


